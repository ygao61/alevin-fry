//! Per-transcript binding-affinity tracks for the Forseti resolver.
//!
//! The affinity of a 30-mer depends only on (transcript, strand, position), so
//! it is computed once per process and shared by all workers, keeping only the
//! "hot" positions (30-mers with a run of >= 6 A; every other position has
//! affinity 0). Scoring a candidate window is a range scan over its hot
//! positions.
//!
//! Exactness: each scored term is `ln(aff * frag + EPS) >= ln(EPS)` (both
//! factors are >= 0; the spline table is clamped at load) and a cold position
//! contributes exactly `ln(EPS)`, so the maximum over hot positions equals the
//! maximum over all positions whenever a hot position exists -- the condition
//! under which the per-window scorer produced a score at all. Affinities are
//! stored as `f32`, lossless for the MLP's `f32` sigmoid output. The frozen
//! per-window scorer (`forseti_reference.rs`) and the `forseti-shadow` feature
//! check this bit-for-bit.
//!
//! Storage: spliced transcripts get one whole-length track; unspliced ones are
//! split into [`UNSPLICED_BLOCK`]-bp blocks built on first use (1024 bp was the
//! fastest of 4096/1024/512 at equal memory on pbmc_1k/10k; see
//! docs/forseti_perf_review_2026-08-19.md, item 2.1). A transcript that is
//! never a candidate costs one empty slot.

use crate::forseti::compute_has_6a;
use crate::mlp_spline::NativeMlp;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

/// k-mer length the MLP was trained on.
pub const K: usize = 30;
/// A 30-mer is "hot" (can have a non-zero affinity) iff it contains a run of
/// at least this many consecutive `A`.
pub const MIN_A_RUN: usize = 6;
/// Number of virtual poly-A-extended tail 30-mers per transcript: window `j`
/// is the last `29 - j` transcript bases followed by `j + 1` `A`s. The scorer
/// never asks for more than 15 (beyond that a window is treated as pure
/// poly-A tail with affinity 1.0).
pub const TAIL_WINDOWS: usize = 15;
/// Default block length (in transcript positions) for lazily built unspliced
/// tracks; `FORSETI_U_BLOCK=<n>` overrides it (experiments only).
pub const UNSPLICED_BLOCK: usize = 1024;

fn unspliced_block() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("FORSETI_U_BLOCK")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n >= K)
            .unwrap_or(UNSPLICED_BLOCK)
    })
}
/// Splicing-status byte (from the 3-column t2g) that selects blocked storage.
pub const UNSPLICED_STATUS: u8 = b'U';

// Affinity post-processing constants, kept next to the builder that applies them.
/// Multiplicative discount applied to the raw MLP output.
pub const DISCOUNT_PERC: f64 = 1.0;
/// Raw affinities at or below this are set to 0.
pub const BINDING_AFFINITY_THRESHOLD: f64 = 0.0;

// ----------------------------------------------------------------- stats --
static TX_TOUCHED: AtomicU64 = AtomicU64::new(0);
static BLOCKS_BUILT: AtomicU64 = AtomicU64::new(0);
static TAILS_BUILT: AtomicU64 = AtomicU64::new(0);
static FWD_ENTRIES: AtomicU64 = AtomicU64::new(0);
static RC_ENTRIES: AtomicU64 = AtomicU64::new(0);
static POSITIONS_SCANNED: AtomicU64 = AtomicU64::new(0);
static MLP_EVALS: AtomicU64 = AtomicU64::new(0);
static BUILD_NANOS: AtomicU64 = AtomicU64::new(0);
static QUERIES: AtomicU64 = AtomicU64::new(0);
static USED_ENTRIES: AtomicU64 = AtomicU64::new(0);

/// Snapshot of the track-building counters.
#[derive(Debug, Clone, Copy)]
pub struct TrackStats {
    /// transcripts that received a slot (were a candidate at least once)
    pub tx_touched: u64,
    /// (strand-pair) blocks built; == tx_touched for spliced-only data
    pub blocks_built: u64,
    pub tails_built: u64,
    pub fwd_entries: u64,
    pub rc_entries: u64,
    /// transcript positions scanned by the 6A pass while building
    pub positions_scanned: u64,
    /// MLP forward passes run while building
    pub mlp_evals: u64,
    /// CPU seconds spent building (summed over threads)
    pub build_secs: f64,
    /// window queries answered (fwd + rc)
    pub queries: u64,
    /// hot entries handed to the scorer, summed over queries
    pub used_entries: u64,
    /// effective unspliced block size
    pub u_block: usize,
}

impl TrackStats {
    /// Approximate resident bytes of the hot-position arrays.
    pub fn entry_bytes(&self) -> u64 {
        (self.fwd_entries + self.rc_entries) * (4 + 4)
    }
}

pub fn track_stats() -> TrackStats {
    TrackStats {
        tx_touched: TX_TOUCHED.load(Ordering::Relaxed),
        blocks_built: BLOCKS_BUILT.load(Ordering::Relaxed),
        tails_built: TAILS_BUILT.load(Ordering::Relaxed),
        fwd_entries: FWD_ENTRIES.load(Ordering::Relaxed),
        rc_entries: RC_ENTRIES.load(Ordering::Relaxed),
        positions_scanned: POSITIONS_SCANNED.load(Ordering::Relaxed),
        mlp_evals: MLP_EVALS.load(Ordering::Relaxed),
        build_secs: BUILD_NANOS.load(Ordering::Relaxed) as f64 * 1e-9,
        queries: QUERIES.load(Ordering::Relaxed),
        used_entries: USED_ENTRIES.load(Ordering::Relaxed),
        u_block: unspliced_block(),
    }
}

// ------------------------------------------------------------ affinity ---

/// Post-processing of one raw MLP output for the 30-mer `window`: all-`A`
/// -> 1.0; otherwise the discounted value, floored to 0 at the threshold.
/// `f32` is lossless for these values.
#[inline]
pub fn finalize_affinity(window: &[u8], raw: f64) -> f32 {
    let nonzero = raw * DISCOUNT_PERC;
    let aff = if window.iter().all(|&b| b == b'A') {
        1.0
    } else if nonzero <= BINDING_AFFINITY_THRESHOLD {
        0.0
    } else {
        nonzero
    };
    aff as f32
}

/// Reverse-complement `src` into `dst` (cleared first): `N`/`n` -> `N`,
/// lower case is upper-cased, anything else is a reference-format error.
fn reverse_complement_into(src: &[u8], dst: &mut Vec<u8>) {
    dst.clear();
    dst.reserve(src.len());
    for &b in src.iter().rev() {
        dst.push(match b {
            b'A' | b'a' => b'T',
            b'T' | b't' => b'A',
            b'C' | b'c' => b'G',
            b'G' | b'g' => b'C',
            b'N' | b'n' => b'N',
            other => panic!("Invalid nucleotide found: {}", other as char),
        });
    }
}

// ------------------------------------------------------------- storage ---

/// Hot positions of one strand within one block, sorted by position.
/// `pos` is the 30-mer *start* on the forward strand, or the 30-mer *end*
/// (exclusive, in forward coordinates) for the reverse-complement strand.
#[derive(Default)]
pub struct Strand {
    pos: Vec<u32>,
    aff: Vec<f32>,
}

impl Strand {
    /// Append every (pos, aff) with `lo <= pos <= hi` to `out`.
    #[inline]
    fn collect(&self, lo: u32, hi: u32, out: &mut Vec<(u32, f32)>) {
        let a = self.pos.partition_point(|&p| p < lo);
        let b = self.pos.partition_point(|&p| p <= hi);
        out.extend(self.pos[a..b].iter().copied().zip(self.aff[a..b].iter().copied()));
    }
}

struct Block {
    fwd: Strand,
    rc: Strand,
    /// number of window queries (fwd + rc) answered by this block; used by
    /// `reuse_report` to see whether caching a block pays off
    queries: AtomicU32,
}

/// One transcript: its blocks (1 for spliced, ceil(len/UNSPLICED_BLOCK) for
/// unspliced) and its poly-A-extended tail windows, all built on first use.
struct TxSlot {
    blocked: bool,
    block_size: usize,
    blocks: Box<[OnceLock<Box<Block>>]>,
    tail: OnceLock<Box<[f32; TAIL_WINDOWS]>>,
}

impl TxSlot {
    fn new(len: usize, blocked: bool) -> Self {
        // Reverse-strand keys run up to `len` inclusive, hence `len + 1`.
        let block_size = if blocked { unspliced_block() } else { len + 1 };
        let n_blocks = (len + 1).div_ceil(block_size).max(1);
        let blocks: Vec<OnceLock<Box<Block>>> = (0..n_blocks).map(|_| OnceLock::new()).collect();
        TxSlot {
            blocked,
            block_size,
            blocks: blocks.into_boxed_slice(),
            tail: OnceLock::new(),
        }
    }
}

/// Process-wide, lazily filled store of hot-position tracks, indexed by
/// transcript id. Shared by all worker threads through an `Arc`; every
/// block/tail is computed exactly once (`OnceLock`), other threads that need
/// the same block wait for that one computation.
pub struct TrackStore {
    slots: Vec<OnceLock<Box<TxSlot>>>,
    /// splicing status per transcript id (`b'U'` -> blocked storage)
    tx_status: Arc<Vec<u8>>,
    mlp: Arc<NativeMlp>,
}

impl TrackStore {
    pub fn new(ref_count: usize, tx_status: Arc<Vec<u8>>, mlp: Arc<NativeMlp>) -> Self {
        debug_assert_eq!(mlp.k(), K);
        let slots = (0..ref_count).map(|_| OnceLock::new()).collect();
        TrackStore { slots, tx_status, mlp }
    }

    pub fn mlp(&self) -> &NativeMlp {
        &self.mlp
    }

    #[inline]
    fn slot(&self, tid: u32, seq: &[u8]) -> &TxSlot {
        self.slots[tid as usize].get_or_init(|| {
            TX_TOUCHED.fetch_add(1, Ordering::Relaxed);
            let blocked = self
                .tx_status
                .get(tid as usize)
                .map_or(false, |&s| s == UNSPLICED_STATUS);
            Box::new(TxSlot::new(seq.len(), blocked))
        })
    }

    #[inline]
    fn block<'a>(&'a self, slot: &'a TxSlot, b: usize, seq: &[u8]) -> &'a Block {
        slot.blocks[b].get_or_init(|| Box::new(self.build_block(seq, b * slot.block_size, slot.block_size)))
    }

    /// Hot forward-strand 30-mer starts `p` with `p_lo <= p <= p_hi`, in
    /// ascending order, appended to `out` (which is cleared first).
    pub fn fwd_hot(&self, tid: u32, seq: &[u8], p_lo: usize, p_hi: usize, out: &mut Vec<(u32, f32)>) {
        out.clear();
        if p_lo > p_hi {
            return;
        }
        let slot = self.slot(tid, seq);
        let (b0, b1) = (p_lo / slot.block_size, p_hi / slot.block_size);
        for b in b0..=b1.min(slot.blocks.len() - 1) {
            let blk = self.block(slot, b, seq);
            blk.queries.fetch_add(1, Ordering::Relaxed);
            blk.fwd.collect(p_lo as u32, p_hi as u32, out);
        }
        QUERIES.fetch_add(1, Ordering::Relaxed);
        USED_ENTRIES.fetch_add(out.len() as u64, Ordering::Relaxed);
    }

    /// Hot reverse-complement 30-mers keyed by their forward-coordinate end
    /// `q` (the 30-mer is `revcomp(seq[q-30..q])`), `q_lo <= q <= q_hi`,
    /// ascending, appended to `out` (cleared first).
    pub fn rc_hot(&self, tid: u32, seq: &[u8], q_lo: usize, q_hi: usize, out: &mut Vec<(u32, f32)>) {
        out.clear();
        if q_lo > q_hi {
            return;
        }
        let slot = self.slot(tid, seq);
        let (b0, b1) = (q_lo / slot.block_size, q_hi / slot.block_size);
        for b in b0..=b1.min(slot.blocks.len() - 1) {
            let blk = self.block(slot, b, seq);
            blk.queries.fetch_add(1, Ordering::Relaxed);
            blk.rc.collect(q_lo as u32, q_hi as u32, out);
        }
        QUERIES.fetch_add(1, Ordering::Relaxed);
        USED_ENTRIES.fetch_add(out.len() as u64, Ordering::Relaxed);
    }

    /// Affinities of the poly-A-extended tail windows `j = 0..TAIL_WINDOWS`
    /// (window `j` = last `29 - j` bases + `j + 1` `A`s); 0.0 where the window
    /// has no 6A run. Only meaningful for transcripts of >= 29 bases (the
    /// scorer never consults the tail otherwise).
    pub fn tail(&self, tid: u32, seq: &[u8]) -> &[f32; TAIL_WINDOWS] {
        let slot = self.slot(tid, seq);
        slot.tail.get_or_init(|| Box::new(self.build_tail(seq)))
    }

    // ------------------------------------------------------- builders ---

    fn build_block(&self, seq: &[u8], start: usize, block_size: usize) -> Block {
        let t0 = Instant::now();
        let len = seq.len();
        let mut hbuf = vec![0f32; self.mlp.hidden()];
        let mut evals = 0u64;
        let mut scanned = 0u64;

        // ---- forward strand: starts p in [start, end_p) with p <= len - K
        let mut fwd = Strand::default();
        let end_p = (start + block_size).min(len.saturating_sub(K - 1)); // exclusive
        if start < end_p {
            let sub = &seq[start..(end_p + K - 1).min(len)];
            let hot = compute_has_6a(sub, K, MIN_A_RUN);
            scanned += sub.len() as u64;
            for i in 0..(end_p - start) {
                if hot[i] {
                    let raw = self.mlp.predict_at(sub, i, &mut hbuf);
                    evals += 1;
                    fwd.pos.push((start + i) as u32);
                    fwd.aff.push(finalize_affinity(&sub[i..i + K], raw));
                }
            }
        }

        // ---- reverse strand: ends q in [start, start + block_size) with K <= q <= len
        let mut rc = Strand::default();
        let q_lo = start.max(K);
        let q_hi = (start + block_size - 1).min(len); // inclusive
        if q_lo <= q_hi {
            let sub = &seq[q_lo - K..q_hi];
            let mut rcbuf = Vec::new();
            reverse_complement_into(sub, &mut rcbuf);
            let hot = compute_has_6a(&rcbuf, K, MIN_A_RUN);
            scanned += rcbuf.len() as u64;
            // rc window i = revcomp(sub[L-i-K .. L-i]) ends at forward q = q_hi - i
            for q in q_lo..=q_hi {
                let i = q_hi - q;
                if hot[i] {
                    let raw = self.mlp.predict_at(&rcbuf, i, &mut hbuf);
                    evals += 1;
                    rc.pos.push(q as u32);
                    rc.aff.push(finalize_affinity(&rcbuf[i..i + K], raw));
                }
            }
        }

        BLOCKS_BUILT.fetch_add(1, Ordering::Relaxed);
        FWD_ENTRIES.fetch_add(fwd.pos.len() as u64, Ordering::Relaxed);
        RC_ENTRIES.fetch_add(rc.pos.len() as u64, Ordering::Relaxed);
        POSITIONS_SCANNED.fetch_add(scanned, Ordering::Relaxed);
        MLP_EVALS.fetch_add(evals, Ordering::Relaxed);
        BUILD_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        fwd.pos.shrink_to_fit();
        fwd.aff.shrink_to_fit();
        rc.pos.shrink_to_fit();
        rc.aff.shrink_to_fit();
        Block { fwd, rc, queries: AtomicU32::new(0) }
    }

    /// Reuse statistics per storage class (spliced whole-transcript tracks vs
    /// unspliced blocks): how many blocks were queried once, twice, ..., how
    /// many entries they hold, and what share of all queries the reused ones
    /// answered. Lines are ready to log; `tsv` (if given) gets one row per
    /// built block: tid, class, block index, queries, fwd entries, rc entries.
    pub fn reuse_report(&self, tsv: Option<&std::path::Path>) -> Vec<String> {
        const BINS: [(u32, u32, &str); 8] = [
            (1, 1, "1"), (2, 2, "2"), (3, 9, "3-9"), (10, 99, "10-99"),
            (100, 199, "100-199"), (200, 499, "200-499"), (500, 999, "500-999"), (1000, u32::MAX, ">=1000"),
        ];
        // [class][bin] -> (blocks, entries, queries)
        let mut acc = [[(0u64, 0u64, 0u64); 8]; 2];
        let mut w = tsv.map(|p| std::io::BufWriter::new(std::fs::File::create(p).expect("track reuse tsv")));
        use std::io::Write;
        if let Some(w) = w.as_mut() {
            writeln!(w, "tid\tclass\tblock\tqueries\tfwd_entries\trc_entries").ok();
        }
        for (tid, slot) in self.slots.iter().enumerate() {
            let Some(slot) = slot.get() else { continue };
            let class = slot.blocked as usize;
            for (bi, b) in slot.blocks.iter().enumerate() {
                let Some(b) = b.get() else { continue };
                let q = b.queries.load(Ordering::Relaxed);
                let e = (b.fwd.pos.len() + b.rc.pos.len()) as u64;
                let bin = BINS.iter().position(|&(lo, hi, _)| q >= lo && q <= hi).unwrap_or(0);
                let a = &mut acc[class][bin];
                a.0 += 1;
                a.1 += e;
                a.2 += q as u64;
                if let Some(w) = w.as_mut() {
                    writeln!(w, "{}\t{}\t{}\t{}\t{}\t{}", tid, if slot.blocked { "U" } else { "S" }, bi, q, b.fwd.pos.len(), b.rc.pos.len()).ok();
                }
            }
        }
        let mut lines = Vec::new();
        for (class, name) in [(0usize, "S whole-transcript tracks"), (1usize, "U blocks")] {
            let tot: (u64, u64, u64) = acc[class].iter().fold((0, 0, 0), |s, a| (s.0 + a.0, s.1 + a.1, s.2 + a.2));
            lines.push(format!(
                "Forseti track reuse, {}: {} built, {:.1} MB entries, {} queries",
                name, tot.0, tot.1 as f64 * 8.0 / 1e6, tot.2
            ));
            for (i, (_, _, label)) in BINS.iter().enumerate() {
                let a = acc[class][i];
                lines.push(format!(
                    "    queried {:>7}x: {:>9} blocks ({:>5.1}%), {:>7.1} MB ({:>5.1}%), {:>5.1}% of queries",
                    label, a.0, pct(a.0, tot.0), a.1 as f64 * 8.0 / 1e6, pct(a.1, tot.1), pct(a.2, tot.2)
                ));
            }
        }
        lines
    }

    fn build_tail(&self, seq: &[u8]) -> [f32; TAIL_WINDOWS] {
        let t0 = Instant::now();
        let mut out = [0f32; TAIL_WINDOWS];
        let len = seq.len();
        if len >= K - 1 {
            // last 29 bases + 15 A's: window j covers bytes [j, j + 30)
            let mut buf = Vec::with_capacity(K - 1 + TAIL_WINDOWS);
            buf.extend_from_slice(&seq[len - (K - 1)..]);
            buf.extend(std::iter::repeat(b'A').take(TAIL_WINDOWS));
            let hot = compute_has_6a(&buf, K, MIN_A_RUN);
            let mut hbuf = vec![0f32; self.mlp.hidden()];
            let mut evals = 0u64;
            for j in 0..TAIL_WINDOWS {
                if hot[j] {
                    let raw = self.mlp.predict_at(&buf, j, &mut hbuf);
                    evals += 1;
                    out[j] = finalize_affinity(&buf[j..j + K], raw);
                }
            }
            MLP_EVALS.fetch_add(evals, Ordering::Relaxed);
        }
        TAILS_BUILT.fetch_add(1, Ordering::Relaxed);
        BUILD_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        out
    }
}

fn pct(a: u64, b: u64) -> f64 {
    if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 }
}
