//! On-demand access to the spliceu transcript sequences.
//!
//! The affinity tracks (`track.rs`) read a transcript's bytes once, while a
//! block or tail is built, so the spliceu FASTA (23 GB for human GENCODE) no
//! longer has to be resident. [`FaiSeqStore`] keeps only the `.fai` index in
//! memory and serves each request with a positional read; the data stays in
//! the shared, evictable kernel page cache rather than in this process' RSS.
//! Random 1 kb reads from a cold network file system are slow: stage the
//! FASTA to local disk (or read it through once) before a run.
//!
//! Requires `<fasta>.fai` (`samtools faidx`). Multi-line records are handled
//! through the fai line geometry; single-line records (roers output) take one
//! read per request.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Something that can hand out transcript sequence by id.
pub trait SeqSource: Send + Sync {
    /// Length of transcript `tid`, `None` if it is not in the store.
    fn len(&self, tid: u32) -> Option<usize>;
    /// Copy `seq[start..end)` of transcript `tid` into `buf` (cleared first).
    /// `end <= len(tid)` is the caller's responsibility.
    fn fetch(&self, tid: u32, start: usize, end: usize, buf: &mut Vec<u8>);
}

static FETCHES: AtomicU64 = AtomicU64::new(0);
static FETCHED_BYTES: AtomicU64 = AtomicU64::new(0);

/// (number of fetches, bytes copied) served so far by any `FaiSeqStore`.
pub fn fetch_stats() -> (u64, u64) {
    (FETCHES.load(Ordering::Relaxed), FETCHED_BYTES.load(Ordering::Relaxed))
}

#[derive(Clone, Copy)]
struct FaiRec {
    offset: u64,
    len: u32,
    linebases: u32,
    linewidth: u32,
}

/// `.fai`-indexed FASTA read on demand with positional reads.
pub struct FaiSeqStore {
    file: File,
    /// indexed by transcript id (RAD reference order)
    recs: Vec<Option<FaiRec>>,
    n_present: usize,
}

impl FaiSeqStore {
    /// Open `fasta` and `fasta.fai`; records are keyed by their position in
    /// `ref_names` (the RAD header order), FASTA names not in `ref_names` are
    /// ignored and `ref_names` without a FASTA record report `len == None`.
    pub fn open(fasta: &Path, ref_names: &[String]) -> Result<Self> {
        let fai_path = {
            let mut s = fasta.as_os_str().to_owned();
            s.push(".fai");
            std::path::PathBuf::from(s)
        };
        if fasta.extension().is_some_and(|e| e.eq_ignore_ascii_case("gz")) {
            bail!(
                "{:?} looks gzip-compressed; forseti reads transcript sequence by byte offset \
                 through the .fai index, which needs the uncompressed FASTA",
                fasta
            );
        }
        let fai = std::fs::read_to_string(&fai_path).with_context(|| {
            format!(
                "could not read the FASTA index {:?}; create it with `samtools faidx {:?}`",
                fai_path, fasta
            )
        })?;
        // A .fai older than its FASTA describes some other file: offsets would
        // silently read wrong bytes.
        if let (Ok(fm), Ok(im)) = (std::fs::metadata(fasta), std::fs::metadata(&fai_path)) {
            if let (Ok(ft), Ok(it)) = (fm.modified(), im.modified()) {
                if ft > it {
                    bail!(
                        "{:?} is newer than its index {:?}; rebuild the index with `samtools faidx {:?}`",
                        fasta, fai_path, fasta
                    );
                }
            }
        }
        let name_to_id: HashMap<&str, usize> = ref_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        let mut recs: Vec<Option<FaiRec>> = vec![None; ref_names.len()];
        let mut n_present = 0usize;
        for (ln, line) in fai.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let mut it = line.split('\t');
            let name = it.next().unwrap_or("");
            let parse = |s: Option<&str>, what: &str| -> Result<u64> {
                s.and_then(|v| v.trim().parse::<u64>().ok())
                    .with_context(|| format!("{:?} line {}: bad {} field", fai_path, ln + 1, what))
            };
            let len = parse(it.next(), "length")?;
            let offset = parse(it.next(), "offset")?;
            let linebases = parse(it.next(), "linebases")?;
            let linewidth = parse(it.next(), "linewidth")?;
            // samtools writes linebases = linewidth = 0 for an empty record
            if (linebases == 0 && len > 0) || linewidth < linebases {
                bail!("{:?} line {}: invalid line geometry", fai_path, ln + 1);
            }
            if let Some(&id) = name_to_id.get(name) {
                recs[id] = Some(FaiRec {
                    offset,
                    len: u32::try_from(len).context("transcript longer than u32::MAX")?,
                    linebases: linebases as u32,
                    linewidth: linewidth as u32,
                });
                n_present += 1;
            }
        }
        let file = File::open(fasta).with_context(|| format!("failed to open spliceu fasta {:?}", fasta))?;
        Ok(FaiSeqStore { file, recs, n_present })
    }

    /// Number of RAD references that have a FASTA record.
    pub fn n_present(&self) -> usize {
        self.n_present
    }

    /// Up to `limit` names of RAD references without a FASTA record.
    pub fn missing_names(&self, ref_names: &[String], limit: usize) -> Vec<String> {
        self.recs
            .iter()
            .zip(ref_names)
            .filter(|(r, _)| r.is_none())
            .map(|(_, n)| n.clone())
            .take(limit)
            .collect()
    }
}

impl SeqSource for FaiSeqStore {
    fn len(&self, tid: u32) -> Option<usize> {
        self.recs.get(tid as usize).and_then(|r| r.map(|r| r.len as usize))
    }

    fn fetch(&self, tid: u32, start: usize, end: usize, buf: &mut Vec<u8>) {
        buf.clear();
        let Some(r) = self.recs.get(tid as usize).copied().flatten() else { return };
        debug_assert!(end <= r.len as usize && start <= end);
        let want = end - start;
        buf.reserve(want);
        let (lb, lw) = (r.linebases as usize, r.linewidth as usize);
        if lb as u32 + 1 == r.linewidth || lb == lw {
            // fast path: the requested range lies on one line (always true for
            // single-line records); one read, no newline stripping needed
            if start / lb == (end.max(1) - 1) / lb {
                let off = r.offset + (start / lb * lw + start % lb) as u64;
                let old = buf.len();
                buf.resize(old + want, 0);
                self.file
                    .read_exact_at(&mut buf[old..], off)
                    .expect("spliceu fasta: read failed (truncated file or stale .fai?)");
                FETCHES.fetch_add(1, Ordering::Relaxed);
                FETCHED_BYTES.fetch_add(want as u64, Ordering::Relaxed);
                return;
            }
        }
        // general path: walk line by line
        let mut pos = start;
        let mut tmp = Vec::new();
        while pos < end {
            let line = pos / lb;
            let col = pos % lb;
            let take = (lb - col).min(end - pos);
            let off = r.offset + (line * lw + col) as u64;
            tmp.resize(take, 0);
            self.file
                .read_exact_at(&mut tmp, off)
                .expect("spliceu fasta: read failed (truncated file or stale .fai?)");
            buf.extend_from_slice(&tmp);
            pos += take;
        }
        FETCHES.fetch_add(1, Ordering::Relaxed);
        FETCHED_BYTES.fetch_add(want as u64, Ordering::Relaxed);
    }
}

/// In-memory sequences (tests, and the reference path of shadow builds).
pub struct MemSeqStore(pub HashMap<u32, Vec<u8>>);

impl SeqSource for MemSeqStore {
    fn len(&self, tid: u32) -> Option<usize> {
        self.0.get(&tid).map(|s| s.len())
    }
    fn fetch(&self, tid: u32, start: usize, end: usize, buf: &mut Vec<u8>) {
        buf.clear();
        if let Some(s) = self.0.get(&tid) {
            buf.extend_from_slice(&s[start..end]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_fasta(dir: &Path, wrap: Option<usize>, seqs: &[(&str, &str)]) -> std::path::PathBuf {
        let fa = dir.join(format!("t_{}.fa", wrap.unwrap_or(0)));
        let mut f = File::create(&fa).unwrap();
        let mut fai = String::new();
        let mut off = 0u64;
        for (name, seq) in seqs {
            let header = format!(">{} some description\n", name);
            f.write_all(header.as_bytes()).unwrap();
            off += header.len() as u64;
            let lb = wrap.unwrap_or(seq.len().max(1));
            fai.push_str(&format!("{}\t{}\t{}\t{}\t{}\n", name, seq.len(), off, lb, lb + 1));
            for chunk in seq.as_bytes().chunks(lb) {
                f.write_all(chunk).unwrap();
                f.write_all(b"\n").unwrap();
                off += chunk.len() as u64 + 1;
            }
        }
        let mut p = fa.as_os_str().to_owned();
        p.push(".fai");
        std::fs::write(p, fai).unwrap();
        fa
    }

    #[test]
    fn fai_store_matches_memory_store() {
        let dir = std::env::temp_dir().join(format!("seqstore_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let s1 = "ACGTNACGTTTTTTAAAAAACCCGGGTTTACGATCGATCGATCGGGGGAAAAAATTTTTCCCCAGT".repeat(7);
        let s2 = "AAAAAAAAAACGTACGT".to_string();
        let s3 = "N".to_string();
        let seqs = [("tx0", s1.as_str()), ("tx1", s2.as_str()), ("tx2", s3.as_str())];
        let names: Vec<String> = ["tx0", "tx1", "tx2", "missing"].iter().map(|s| s.to_string()).collect();
        let mut mem = HashMap::new();
        for (i, (_, s)) in seqs.iter().enumerate() {
            mem.insert(i as u32, s.as_bytes().to_vec());
        }
        let mem = MemSeqStore(mem);
        for wrap in [None, Some(60), Some(7)] {
            let fa = write_fasta(&dir, wrap, &seqs);
            let st = FaiSeqStore::open(&fa, &names).unwrap();
            assert_eq!(st.n_present(), 3);
            assert_eq!(st.len(3), None);
            let (mut a, mut b) = (Vec::new(), Vec::new());
            for tid in 0..3u32 {
                let l = st.len(tid).unwrap();
                assert_eq!(Some(l), mem.len(tid));
                for start in 0..l {
                    for end in [start, start + 1, (start + 29).min(l), (start + 31).min(l), l] {
                        if end < start {
                            continue;
                        }
                        st.fetch(tid, start, end, &mut a);
                        mem.fetch(tid, start, end, &mut b);
                        assert_eq!(a, b, "wrap {:?} tid {} [{}..{})", wrap, tid, start, end);
                    }
                }
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
