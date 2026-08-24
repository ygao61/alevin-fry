#[cfg(feature = "forseti-shadow")]
use crate::mlp_spline::NativeMlp;
use crate::track::TrackStore;
use anyhow::{bail, Context, Result};
use ndarray::prelude::*;
use ndarray::s;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicU64, Ordering};

type CoveringTxpId = u32;
/// RAD alignment tuple: (is_forward, ref_start)
type AlgnTuple = (bool, u32);
/// One entry per `check_mcc_list` element, in the same order, so the position in
/// this Vec *is* the MCC index reported back to the caller. A Vec (rather than a
/// HashMap keyed by that index) keeps candidate iteration order fixed across runs.
pub type ForsetiCheckingList = Vec<(CoveringTxpId, Vec<AlgnTuple>)>;

/// Allocation-free 6A detection. For each k-mer window start `i` in `[0, len-k]`,
/// returns whether `bytes[i..i+k]` contains a run of >= `min_run` consecutive b'A'.
/// O(len) via a difference array; bit-equivalent to the old per-k-mer
/// `memmem::find(window, "A"*min_run).is_some()` but with zero String allocations.
pub(crate) fn compute_has_6a(bytes: &[u8], k: usize, min_run: usize) -> Array1<bool> {
    if bytes.len() < k {
        return Array1::from(Vec::<bool>::new());
    }
    let n = bytes.len() - k + 1; // number of k-mer windows
    let mut diff = vec![0i32; n + 1];
    let mut run = 0usize;
    for p in 0..bytes.len() {
        if bytes[p] == b'A' {
            run += 1;
        } else {
            run = 0;
        }
        if run >= min_run {
            // a min_run-A substring ends at p, starts at s = p + 1 - min_run.
            // window start i contains it iff i <= s and s + min_run <= i + k,
            // i.e. i in [ (p+1) - k , s ] (clamped to valid window starts).
            let s = p + 1 - min_run;
            let lo = (p + 1).saturating_sub(k);
            let hi = s.min(n - 1);
            if lo <= hi {
                diff[lo] += 1;
                diff[hi + 1] -= 1;
            }
        }
    }
    let mut hot = vec![false; n];
    let mut acc = 0i32;
    for i in 0..n {
        acc += diff[i];
        hot[i] = acc > 0;
    }
    Array1::from(hot)
}

pub fn build_status_lookup(
    t2g_path: &std::path::PathBuf,
    ref_names: &[String],
) -> Result<Vec<u8>> {
    // 1. Map name to tx_id (index) for O(1) translation during parsing
    let name_to_id: HashMap<&str, usize> = ref_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    let file = File::open(t2g_path)
        .with_context(|| format!("Could not open T2G file: {:?}", t2g_path))?;
    let reader = BufReader::new(file);

    // Initialize with a dummy byte (e.g., 0)
    let mut status_lookup = vec![0u8; ref_names.len()];

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() { continue; }

        // Split by Tab since it's a TSV
        let cols: Vec<&str> = line.split('\t').collect();

        // 2. Column count check
        if cols.len() < 3 {
            bail!(
                "Error: forseti need 3 col t2g (contain splicing status). \
                 Found {} columns at line {}. Path: {:?}",
                cols.len(),
                line_idx + 1,
                t2g_path
            );
        }

        let tx_name = cols[0];
        let status_char = cols[2].as_bytes()[0]; // Take 'S', 'U', or 'T'

        // 3. Store status using tx_id as the index
        if let Some(&tx_id) = name_to_id.get(tx_name) {
            status_lookup[tx_id] = status_char;
        }
    }

    Ok(status_lookup)
}

/// Number of (candidate lists, scored candidates, mismatching candidates)
/// seen by the shadow comparison; all zero unless built with
/// `--features forseti-shadow`.
pub static SHADOW_LISTS: AtomicU64 = AtomicU64::new(0);
pub static SHADOW_CANDIDATES: AtomicU64 = AtomicU64::new(0);
pub static SHADOW_MISMATCHES: AtomicU64 = AtomicU64::new(0);

pub fn shadow_stats() -> (u64, u64, u64) {
    (
        SHADOW_LISTS.load(Ordering::Relaxed),
        SHADOW_CANDIDATES.load(Ordering::Relaxed),
        SHADOW_MISMATCHES.load(Ordering::Relaxed),
    )
}

/// Shadow mode: re-score the same candidate list with the frozen per-window
/// scorer (`forseti_reference`) and compare every score bit-for-bit.
/// Mismatches are counted; the first few are printed with enough context to
/// replay them.
#[cfg(feature = "forseti-shadow")]
fn shadow_check(
    forseti_checking_list: &ForsetiCheckingList,
    ref_names: &[String],
    spliceu_txome: &HashMap<u32, Vec<u8>>,
    spline_lookup: &Array1<f64>,
    mlp: &NativeMlp,
    read_length: u16,
    max_frag_len: u16,
    scored: &[(u16, f64)],
) {
    const PRINT_LIMIT: u64 = 20;
    let want = match crate::forseti_reference::reference_score_candidates(
        forseti_checking_list,
        ref_names,
        spliceu_txome,
        spline_lookup,
        mlp,
        read_length,
        max_frag_len,
    ) {
        Ok(w) => w,
        Err(e) => {
            let n = SHADOW_MISMATCHES.fetch_add(1, Ordering::Relaxed);
            if n < PRINT_LIMIT {
                eprintln!("[forseti-shadow] reference errored: {e}; list {:?}", forseti_checking_list);
            }
            return;
        }
    };
    SHADOW_LISTS.fetch_add(1, Ordering::Relaxed);
    SHADOW_CANDIDATES.fetch_add(scored.len() as u64, Ordering::Relaxed);
    let same = scored.len() == want.len()
        && scored
            .iter()
            .zip(want.iter())
            .all(|(&(gi, gs), &(wi, ws))| gi == wi && gs.to_bits() == ws.to_bits());
    if !same {
        let n = SHADOW_MISMATCHES.fetch_add(1, Ordering::Relaxed);
        if n < PRINT_LIMIT {
            eprintln!(
                "[forseti-shadow] MISMATCH #{}: rl {} mfl {}\n  got  {:?}\n  want {:?}\n  list {:?}",
                n + 1, read_length, max_frag_len, scored, want, forseti_checking_list
            );
        }
    }
}

pub fn forseti_for_multi_best(
    forseti_checking_list: &ForsetiCheckingList,
    ref_names: &[String],
    spline_lookup: &Array1<f64>,
    tracks: &TrackStore,
    read_length: u16,
    max_frag_len: u16,
) -> Result<Vec<u16>> {
    let scored = forseti_score_candidates(
        forseti_checking_list,
        ref_names,
        spline_lookup,
        tracks,
        read_length,
        max_frag_len,
    )?;
    #[cfg(feature = "forseti-shadow")]
    shadow_check(
        forseti_checking_list,
        ref_names,
        tracks.shadow_txome().expect("shadow build: TrackStore::with_shadow_txome not set"),
        spline_lookup,
        tracks.mlp(),
        read_length,
        max_frag_len,
        &scored,
    );
    Ok(select_best_mcc_indices(&scored))
}

/// Score every candidate (MCC, transcript) pair in `forseti_checking_list`.
///
/// Returns `(mcc_idx, normalised max joint log-probability)` in check-list
/// order; a candidate that reaches the end of the loop without a usable
/// window is recorded with `f64::NEG_INFINITY`, candidates skipped by an
/// early `continue` are not recorded at all.
///
/// Affinities come from the per-transcript tracks (`track.rs`) and each
/// window is scored over its hot positions only; the `track` module docs
/// explain why this equals the per-window result. `forseti_reference` keeps
/// the per-window scorer; `production_scoring_matches_frozen_reference` and
/// the `forseti-shadow` feature check bit-equality.
pub(crate) fn forseti_score_candidates(
    forseti_checking_list: &ForsetiCheckingList,
    ref_names: &[String],
    spline_lookup: &Array1<f64>,
    tracks: &TrackStore,
    read_length: u16,
    max_frag_len: u16,
) -> Result<Vec<(u16, f64)>> {
    let polya_tail_len = 200;
    let max_frag_len = max_frag_len as usize;

    // Reusable buffers to reduce per-MCC allocations.
    let mut ref_start_list: Vec<usize> = Vec::new();
    let mut ref_end_list: Vec<usize> = Vec::new();
    let mut tail_sum_log_probs: Vec<f64> = Vec::new();
    // hot positions of the current window, from the track store
    let mut hot: Vec<(u32, f32)> = Vec::new();
    const EPS: f64 = 1e-12;
    // Every "affinity == 0" term is ln(0*frag_prob + EPS) = ln(EPS), a constant.
    let ln_eps = EPS.ln();
    // Per-candidate scores; the winner set is chosen by the caller.
    let mut scored: Vec<(u16, f64)> = Vec::new();

    // Candidates are visited in check_mcc_list order (mcc_idx ascending), so the
    // order of the returned list is deterministic.
    for (mcc_idx, (covering_txp_id, algn_tuple_list)) in forseti_checking_list.iter().enumerate() {
        let mut norm_sum_joint_prob = f64::NEG_INFINITY;

        let tx_ref_end = match tracks.len(*covering_txp_id) {
            Some(l) => l,
            None => {
                let nm = ref_names
                    .get(*covering_txp_id as usize)
                    .map(|s| s.as_str())
                    .unwrap_or("<unknown>");
                eprintln!("Error: ref_id {} (name {}) not found in spliceu_txome.", covering_txp_id, nm);
                continue;
            }
        };
        let tid = *covering_txp_id;
        let all_forward = algn_tuple_list.iter().all(|algn_tuple| algn_tuple.0);
        let all_reverse = algn_tuple_list.iter().all(|algn_tuple| !algn_tuple.0);

        let invalid_pos_limit: u32 = u32::MAX - read_length as u32 - 50;
        let mut mcc_invalid = false;

        if all_forward {
            ref_start_list.clear();
            ref_start_list.reserve(algn_tuple_list.len());
            for &(_is_forward, ref_start_u32) in algn_tuple_list.iter() {
                // NOTE: Mimic softclip: a start "close to 2^32" is a clipped
                // alignment and is treated as position 0.
                if ref_start_u32 > invalid_pos_limit {
                    ref_start_list.push(0usize);
                } else {
                    let rs = ref_start_u32 as usize;
                    if rs > tx_ref_end {
                        println!("ref_start: {}", rs);
                        eprintln!("Error: ref_start is not close to 2^32, but exceed ref length");
                        mcc_invalid = true;
                        break;
                    }
                    ref_start_list.push(rs);
                }
            }
            if mcc_invalid {
                continue;
            }
            if ref_start_list.is_empty() {
                eprintln!("ref_start_list is empty! Should not happen.");
                continue;
            }
            let n_alns = ref_start_list.len() as f64;

            let overlap_wdow_start = *ref_start_list.iter().max().unwrap();
            let overlap_wdow_end = *ref_start_list.iter().min().unwrap() + max_frag_len;
            if overlap_wdow_start >= overlap_wdow_end {
                continue;
            }

            // Downstream (internal poly-A) arm: 30-mers of
            // ref[ws+1 .. min(we+30, tx_end)), i.e. starts p in [ws+1, e-30].
            if tx_ref_end - overlap_wdow_start > 30 {
                let e = usize::min(overlap_wdow_end + 30, tx_ref_end);
                if e - (overlap_wdow_start + 1) < 30 {
                    // fewer than one 30-mer in the window
                    continue;
                }
                tracks.fwd_hot(tid, overlap_wdow_start + 1, e - 30, &mut hot);
                if !hot.is_empty() {
                    let mut max_sum_log = f64::NEG_INFINITY;
                    for &(p, aff) in hot.iter() {
                        let affinity = aff as f64;
                        let p = p as usize;
                        let mut sum_log_prob = 0.0f64;
                        for &ref_start in &ref_start_list {
                            let frag_prob = spline_lookup[p - ref_start];
                            sum_log_prob += if affinity == 0.0 {
                                ln_eps
                            } else {
                                (affinity * frag_prob + EPS).ln()
                            };
                        }
                        max_sum_log = f64::max(max_sum_log, sum_log_prob);
                    }
                    // make it comparable across different n_alns
                    norm_sum_joint_prob = max_sum_log / n_alns;
                }
            }

            // Tail arm: the window runs past the last 30-mer of the reference, so
            // borrow A's from the poly-A tail (virtual 30-mers = last 29-j bases +
            // j+1 A's, at most 15; beyond that a window counts as pure poly-A).
            // `tx_ref_end >= 30` mirrors the original unsigned `we > tx_end - 30`
            // (a reference shorter than 30 wrapped and never took this arm).
            if tx_ref_end >= 30 && overlap_wdow_end > tx_ref_end - 30 {
                let needed_extra_a_len = 30 + overlap_wdow_end - tx_ref_end;
                let n_tail = needed_extra_a_len.min(15);

                let tx_end_30mer_start = tx_ref_end.saturating_sub(29);
                if overlap_wdow_start >= tx_end_30mer_start {
                    continue; // skip the overhanging cases (drops the candidate, as before)
                }
                let ovlp_wdow_dis_to_tx_end_30mer = tx_end_30mer_start - overlap_wdow_start;
                let ovlp_wdow_length = (overlap_wdow_end - overlap_wdow_start + 1)
                    .min(ovlp_wdow_dis_to_tx_end_30mer + 1 + polya_tail_len);
                let valid_len = ovlp_wdow_length.saturating_sub(ovlp_wdow_dis_to_tx_end_30mer);

                let tail_aff = tracks.tail(tid);
                tail_sum_log_probs.clear();
                tail_sum_log_probs.resize(valid_len, 0.0);
                for &ref_start in &ref_start_list {
                    let prefix_dis = overlap_wdow_start - ref_start;
                    let start_idx = prefix_dis + ovlp_wdow_dis_to_tx_end_30mer;
                    let end_idx = prefix_dis + ovlp_wdow_length;
                    let tail_frag_len_prob = spline_lookup.slice(s![start_idx..end_idx]);
                    for (i, frag_prob) in tail_frag_len_prob.iter().enumerate() {
                        let affinity = if i < n_tail { tail_aff[i] as f64 } else { 1.0 };
                        tail_sum_log_probs[i] += if affinity == 0.0 {
                            ln_eps
                        } else {
                            (affinity * frag_prob + EPS).ln()
                        };
                    }
                }
                let max_log = tail_sum_log_probs
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max);
                let norm_max_tail_log = max_log / n_alns;
                if norm_max_tail_log > norm_sum_joint_prob {
                    norm_sum_joint_prob = norm_max_tail_log;
                }
            }
        } else if all_reverse {
            ref_end_list.clear();
            ref_end_list.reserve(algn_tuple_list.len());
            for &(_is_reverse, ref_start_u32) in algn_tuple_list.iter() {
                let mut ref_start = ref_start_u32 as usize;
                if ref_start_u32 > invalid_pos_limit {
                    // mimic softclip
                    ref_start = 0usize;
                } else if ref_start > tx_ref_end {
                    eprintln!("Error: ref_start is not close to 2^32, but exceed ref length. This should not happen.");
                    mcc_invalid = true;
                    break;
                }
                // clamp to transcript end to avoid panics near the end (or for clipped alignments)
                let ref_end = ref_start
                    .saturating_add(read_length as usize)
                    .min(tx_ref_end);
                ref_end_list.push(ref_end);
            }
            if mcc_invalid {
                continue;
            }
            let n_alns = ref_end_list.len() as f64;

            let overlap_wdow_end = *ref_end_list.iter().min().unwrap();
            let overlap_wdow_start = ref_end_list
                .iter()
                .cloned()
                .max()
                .unwrap()
                .saturating_sub(max_frag_len);
            if overlap_wdow_start >= overlap_wdow_end {
                continue;
            }

            if overlap_wdow_end > 30 {
                // Upstream (antisense) arm: reverse complement of
                // ref[start_pos .. end_pos); rc 30-mer i ends at forward q = end_pos - i,
                // so q runs over [start_pos + 30, end_pos].
                let start_pos = overlap_wdow_start.max(30);
                let end_pos = overlap_wdow_end.min(tx_ref_end);
                if end_pos <= start_pos {
                    continue;
                }
                if end_pos - start_pos >= 30 {
                    tracks.rc_hot(tid, start_pos + 30, end_pos, &mut hot);
                    if !hot.is_empty() {
                        // New: add penalty for antisense reads
                        let anti_sense_penalty = 0.8;
                        let mut max_sum = f64::NEG_INFINITY;
                        for &(q, aff) in hot.iter() {
                            let affinity = aff as f64;
                            let q = q as usize;
                            let mut sum_log_prob = 0.0f64;
                            for &ref_end in &ref_end_list {
                                let frag_prob = spline_lookup[ref_end - q + 1];
                                sum_log_prob += if affinity == 0.0 {
                                    ln_eps
                                } else {
                                    (affinity * frag_prob * anti_sense_penalty + EPS).ln()
                                };
                            }
                            max_sum = f64::max(max_sum, sum_log_prob);
                        }
                        norm_sum_joint_prob = max_sum / n_alns;
                    }
                }
            } else {
                continue;
            }
        } else {
            eprintln!("Error: algn_tuple_list is not all forward or all reverse. This should not happen.");
            continue;
        }

        // Just record the score here; the winner set is picked by the caller.
        scored.push((mcc_idx as u16, norm_sum_joint_prob));
    }

    Ok(scored)
}

/// Pick the winner set from per-candidate scores: the true maximum first,
/// then every candidate within `TIE_EPS` of it, in mcc_idx order.
pub(crate) fn select_best_mcc_indices(scored: &[(u16, f64)]) -> Vec<u16> {
    // Take the true maximum first, then collect every candidate within TIE_EPS
    // of it. The previous single running-max pass was order dependent: "within
    // TIE_EPS" is not transitive, so a candidate that ties against one anchor is
    // discarded against another, and the anchor depended on which candidate the
    // iteration happened to visit first. Two passes make the winner set a
    // function of the scores alone, and its order is mcc_idx ascending.
    const TIE_EPS: f64 = 1e-6;
    let max_score = scored
        .iter()
        .map(|&(_, score)| score)
        .fold(f64::NEG_INFINITY, f64::max);
    if max_score == f64::NEG_INFINITY {
        // No candidate could be scored at all; forseti abstains and the caller
        // keeps the labels it held before calling us.
        return Vec::new();
    }
    scored
        .iter()
        .filter(|&&(_, score)| score != f64::NEG_INFINITY && (max_score - score) <= TIE_EPS)
        .map(|&(mcc_idx, _)| mcc_idx)
        .collect()
}
