//! Frozen reference implementation of the Forseti candidate scoring.
//!
//! Verbatim copy of `forseti::forseti_score_candidates` as of commit 3af81cf:
//! per window, a 6A scan + MLP + log-sum over every 30-mer position. Compiled
//! only for tests and `--features forseti-shadow`, so any rewrite of the
//! scoring can be checked bit-for-bit against the original semantics. Do not
//! "improve" this file: a deliberate change of semantics is its own commit
//! with a before/after comparison, after which this copy is re-frozen.
//!
//! Three deliberate differences from 3af81cf, none changing a value: the
//! candidate index is `u32` instead of `u16` (type only, matching production); the
//! per-thread MLP affinity cache is gone (it stored final affinities, so every
//! hot 30-mer is simply evaluated directly), and the tail-arm test uses
//! `tx_ref_end.wrapping_sub(30)` so that the release-build behaviour (the arm
//! is skipped for references shorter than 30) also holds under overflow checks.

#![allow(dead_code)]

use crate::forseti::ForsetiCheckingList;
use crate::mlp_spline::NativeMlp;
use anyhow::Result;
use ndarray::prelude::*;
use ndarray::s;
use std::collections::HashMap;

fn reverse_complement(seq: &str) -> Result<String, String> {
    let complement = |base: char| match base {
        'A' | 'a' => Ok('T'),
        'T' | 't' => Ok('A'),
        'C' | 'c' => Ok('G'),
        'G' | 'g' => Ok('C'),
        'N' | 'n' => Ok('N'),
        _ => Err(format!("Invalid nucleotide found: {}", base)),
    };

    seq.chars().rev().map(complement).collect() // Collect into a Result<String, _>
}

fn compute_has_6a(bytes: &[u8], k: usize, min_run: usize) -> Array1<bool> {
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

/// Cache-free copy of `process_binding_affinity` (commit 3af81cf): all-A ->
/// 1.0, `<= threshold` -> 0.0, else the discounted MLP output.
fn reference_binding_affinity(
    bytes: &[u8],
    k: usize,
    has_enough_a: &Array1<bool>,
    mlp: &NativeMlp,
    discount_perc: f64,
    binding_affinity_threshold: f64,
) -> Result<Array1<f64>> {
    let indices: Vec<usize> = has_enough_a
        .indexed_iter()
        .filter_map(|(i, &val)| if val { Some(i) } else { None })
        .collect();
    let mut out = Array1::<f64>::zeros(has_enough_a.len());
    if !indices.is_empty() {
        debug_assert_eq!(k, mlp.k());
        let mut nonzero = mlp.predict_starts(bytes, &indices);
        if nonzero.len() != indices.len() {
            return Err(anyhow::anyhow!(
                "Mismatched dimensions between binding affinity and has_enough_a"
            ));
        }
        nonzero *= discount_perc;
        for (j, &start) in indices.iter().enumerate() {
            let aff = if bytes[start..start + k].iter().all(|&b| b == b'A') {
                1.0
            } else if nonzero[j] <= binding_affinity_threshold {
                0.0
            } else {
                nonzero[j]
            };
            out[start] = aff;
        }
    }
    Ok(out)
}

pub(crate) fn reference_score_candidates(
    forseti_checking_list: &ForsetiCheckingList,
    ref_names: &[String],
    spliceu_txome: &HashMap<u32, Vec<u8>>,
    spline_lookup: &Array1<f64>,
    mlp: &NativeMlp,
    read_length: u16,
    max_frag_len: u16,
) -> Result<Vec<(u32, f64)>> {
    // Set up parameters
    let snr_min_size = 6;
    let discount_perc = 1.0_f64;
    let polya_tail_len = 200;
    let max_frag_len = max_frag_len;
    let binding_affinity_threshold = 0.0;
    // 6A detection is done allocation-free in compute_has_6a (uses snr_min_size).

    // Avoid turning stderr into the bottleneck when many MCCs are skipped.
    const INVALID_POS_PRINT_LIMIT: u64 = 2;
    let mut invalid_pos_skipped: u64 = 0;

    // Reusable buffers to reduce per-MCC allocations.
    let mut ref_start_list: Vec<usize> = Vec::new();
    let mut ref_end_list: Vec<usize> = Vec::new();
    // reusable sum buffers for accumulating joint probabilities across alignments
    let mut sum_log_probs: Vec<f64> = Vec::new();
    let mut tail_sum_log_probs: Vec<f64> = Vec::new();
    const EPS: f64 = 1e-12;
    // Every "affinity == 0" term is ln(0*frag_prob + EPS) = ln(EPS), a constant.
    // ~98% of 30-mers have affinity 0, so add this instead of calling the costly .ln().
    let ln_eps = EPS.ln();
    // Per-candidate scores; the winner set is chosen in a second pass below.
    let mut scored: Vec<(u32, f64)> = Vec::new();


    // Candidates are visited in check_mcc_list order (mcc_idx ascending), so the
    // order of the returned winner list is deterministic.
    // algn_tuple_list is direction(fw, reverse) and ref_start
    for (mcc_idx, (covering_txp_id, algn_tuple_list)) in forseti_checking_list.iter().enumerate() {
        let mut norm_sum_joint_prob = f64::NEG_INFINITY;

        let ref_seq_bytes = match spliceu_txome.get(covering_txp_id) {
            Some(seq) => seq.as_slice(),
            None => {
                let nm = ref_names
                    .get(*covering_txp_id as usize)
                    .map(|s| s.as_str())
                    .unwrap_or("<unknown>");
                eprintln!("Error: ref_id {} (name {}) not found in spliceu_txome.", covering_txp_id, nm);
                continue;
            }
        };
        // SAFETY: the spliceu reference is DNA (A/C/G/T/N) -> always valid ASCII -> valid
        // UTF-8, so validation can never fail. `from_utf8` would re-scan the whole
        // transcript (up to ~45 kb) on every candidate (~4% of runtime); skip it.
        let ref_seq = unsafe { std::str::from_utf8_unchecked(ref_seq_bytes) };
        let tx_ref_end = ref_seq.len();
        let all_forward = algn_tuple_list.iter().all(|algn_tuple| algn_tuple.0);
        let all_reverse = algn_tuple_list.iter().all(|algn_tuple| !algn_tuple.0);

        let invalid_pos_limit: u32 = u32::MAX - read_length as u32 - 50;
        let mut mcc_invalid = false;

        if all_forward {
            ref_start_list.clear();
            ref_start_list.reserve(algn_tuple_list.len());
            for &(_is_forward, ref_start_u32) in algn_tuple_list.iter() {
                // NOTE: Mimic softclip
                // TODO:we could have a better way to handle this. if we can use signed int for the ref_start, we could know the length of overhang/clipping.
                if ref_start_u32 > invalid_pos_limit {
                    ref_start_list.push(0 as usize);
                }else{
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
            if mcc_invalid{
                continue;
            }
            if ref_start_list.is_empty() {
                eprintln!("ref_start_list is empty! Should not happen.");
                continue;
            }

            let overlap_wdow_start = *ref_start_list.iter().max().unwrap();
            let overlap_wdow_end = *ref_start_list.iter().min().unwrap() + max_frag_len as usize;
            if overlap_wdow_start >= overlap_wdow_end {
                continue;
            }

            // Downstream processing
            // if we have >30 bases downstream, we want to consider internal polyA sites
            // we use > 30 because the polyA should start one base after the window range start
            if tx_ref_end - overlap_wdow_start > 30 {
                let downstream_seq = &ref_seq
                    [(overlap_wdow_start + 1)..usize::min(overlap_wdow_end + 30, tx_ref_end)];
                let ds_bytes = downstream_seq.as_bytes();
                let has_enough_a = compute_has_6a(ds_bytes, 30, snr_min_size);
                if has_enough_a.is_empty() {
                    continue;
                }
                let n_kmers = has_enough_a.len();
                // if any of the 30mers has enough A, we can process the binding affinity for this mcc
                if has_enough_a.iter().any(|&x| x) {
                    let downstream_binding_affinity = reference_binding_affinity(
                        ds_bytes,
                        30,
                        &has_enough_a,
                        mlp,
                        discount_perc,
                        binding_affinity_threshold,
                    )?;

                    sum_log_probs.clear();
                    sum_log_probs.resize(n_kmers, 0.0);

                    // for each alignment, we compute the prob. distanse = (each algn's start to the overlapped window end), while poly A range is overlap wdow start to end.
                    for &ref_start in &ref_start_list {
                        let prefix_dis = overlap_wdow_start - ref_start;
                        let start_idx = prefix_dis + 1;
                        let end_idx = prefix_dis + n_kmers + 1;
                        let downstream_frag_len_prob = spline_lookup.slice(s![start_idx..end_idx]);
                        for (i, (&affinity, &frag_prob)) in downstream_binding_affinity
                            .iter()
                            .zip(downstream_frag_len_prob.iter())
                            .enumerate()
                        {
                            // avoid log(0), + EPS; affinity==0 (~98% of 30-mers) -> ln(EPS) constant
                            sum_log_probs[i] += if affinity == 0.0 {
                                ln_eps
                            } else {
                                (affinity * frag_prob + EPS).ln()
                            };
                        }
                    }

                    let max_sum_log = sum_log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    // make it comparable across different n_alns
                    norm_sum_joint_prob = max_sum_log / ref_start_list.len() as f64;
                }
            }
            // the downstream window end is beyond the last 30 mer of the ref, we could borrow A from the poly A tail.
            // And, we only compute the case that we consider borrowed A from tail(avoid duplicate computation for cases above, pure internal polyA)
            // The shipped scorer was only ever run in release builds, where this
            // subtraction wraps for transcripts shorter than 30 bp and the arm is
            // skipped; wrapping_sub keeps that behaviour under overflow checks too.
            if overlap_wdow_end > tx_ref_end.wrapping_sub(30) {
                let needed_extra_a_len = 30 + overlap_wdow_end - tx_ref_end;
                // Build tail 30-mers with extra added "A"s from polyA tail
                // at most we add 15 extra A, since when 30mer has >15A, we will consider this as polyA tail. no need to add more& we can be efficient.
                let tail_seq = format!(
                    "{}{}",
                    &ref_seq[tx_ref_end.saturating_sub(30 - 1)..],
                    "A".repeat(needed_extra_a_len.min(15))
                );
                let tail_bytes = tail_seq.as_bytes();
                let has_enough_a = compute_has_6a(tail_bytes, 30, snr_min_size);
                let n_tail = has_enough_a.len();

                let tail_binding_affinity = if has_enough_a.iter().any(|&x| x) {
                    reference_binding_affinity(
                        tail_bytes,
                        30,
                        &has_enough_a,
                        mlp,
                        discount_perc,
                        binding_affinity_threshold,
                    )?
                } else {
                    Array1::zeros(n_tail)
                };

                // Compute joint probabilities for the tail
                let all_a_prob = 1.0;

                // ovlp_wdow_dis_to_tx_end_30mer is the region we already computedin above downstream branch; (overlap_wdow_end  - overlap_wdow_start + 1)is the length of the shared sliding window.
                // let ovlp_wdow_dis_to_tx_end_30mer = tx_ref_end - 30 + 1 - overlap_wdow_start;
                let tx_end_30mer_start = tx_ref_end.saturating_sub(29); // = tx_ref_end - 30 + 1
                if overlap_wdow_start >= tx_end_30mer_start {
                    continue; // skip the overhanging cases
                }
                let ovlp_wdow_dis_to_tx_end_30mer = tx_end_30mer_start - overlap_wdow_start;

                let ovlp_wdow_length = (overlap_wdow_end  - overlap_wdow_start + 1)
                .min(ovlp_wdow_dis_to_tx_end_30mer + 1 + polya_tail_len);
                let valid_len = ovlp_wdow_length.saturating_sub(ovlp_wdow_dis_to_tx_end_30mer);
                tail_sum_log_probs.clear();
                tail_sum_log_probs.resize(valid_len, 0.0);
                // tail_joint_prob length is determined per-alignment
                for &ref_start in &ref_start_list {
                    // here we compute the ovlp_start to the last 30 mer of the ref
                    // because this is the cases we did not covered by the above arm(downstream window end is within the ref)

                    let prefix_dis = overlap_wdow_start - ref_start;
                    let start_idx = prefix_dis + ovlp_wdow_dis_to_tx_end_30mer;
                    // NOTE: predfix_dis is the per algn dis;
                    let end_idx = prefix_dis+ ovlp_wdow_length;

                    let tail_frag_len_prob = spline_lookup.slice(s![start_idx..end_idx]);

                    // Update joint probabilities with tail_binding_affinity
                    for (i, frag_prob) in tail_frag_len_prob.iter().enumerate() {
                        let affinity = if i < n_tail {
                             tail_binding_affinity[i]
                        } else {
                             all_a_prob // All A probability; if we got >15 A, also apply all A probability, as this is more likely to be polyA tail mode, not internal polyA mode.
                        };
                        tail_sum_log_probs[i] += if affinity == 0.0 {
                            ln_eps
                        } else {
                            (affinity * frag_prob + EPS).ln()
                        };
                    }
                }

                // Sum and normalize joint probabilities for the tail
                let max_log = tail_sum_log_probs
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);

                let norm_max_tail_log = max_log / ref_start_list.len() as f64;

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
                    ref_start = 0 as usize;
                }else if ref_start > tx_ref_end{
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

            if mcc_invalid{
                continue;
            }

            let overlap_wdow_end = *ref_end_list.iter().min().unwrap();
            let overlap_wdow_start = ref_end_list
                .iter()
                .cloned()
                .max()
                .unwrap()
                .saturating_sub(max_frag_len as usize);
            if overlap_wdow_start >= overlap_wdow_end {
                continue;
            }

            if overlap_wdow_end > 30 {
                let start_pos = overlap_wdow_start.max(30);
                let end_pos = overlap_wdow_end.min(tx_ref_end);
                if end_pos <= start_pos {
                    continue;
                }
                let seq_slice = &ref_seq[start_pos..end_pos];
                let rev_comp_seq = reverse_complement(seq_slice).unwrap();

                // Build kmers
                let rc_bytes = rev_comp_seq.as_bytes();
                let has_enough_a = compute_has_6a(rc_bytes, 30, snr_min_size);
                let n_up = has_enough_a.len();

                if has_enough_a.iter().any(|&x| x) {
                    let upstream_binding_affinity = reference_binding_affinity(
                        rc_bytes,
                        30,
                        &has_enough_a,
                        mlp,
                        discount_perc,
                        binding_affinity_threshold,
                    )?;
                    // New: add penalty for antisense reads
                    let anti_sense_penalty = 0.8;
                    // For each alignment, compute fragment length probabilities
                    sum_log_probs.clear();
                    sum_log_probs.resize(n_up, 0.0);
                    for &ref_end in &ref_end_list {
                        let suffix_dis = ref_end - overlap_wdow_end;
                        let start_idx = suffix_dis + 1;
                        let end_idx = n_up + suffix_dis + 1;
                        let upstream_frag_len_prob = spline_lookup.slice(s![start_idx..end_idx]);
                        for (i, (&affinity, &frag_prob)) in upstream_binding_affinity
                            .iter()
                            .zip(upstream_frag_len_prob.iter())
                            .enumerate()
                        {
                            sum_log_probs[i] += if affinity == 0.0 {
                                ln_eps
                            } else {
                                (affinity * frag_prob * anti_sense_penalty + EPS).ln()
                            };
                        }
                    }
                    let max_sum = sum_log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    norm_sum_joint_prob = max_sum / ref_end_list.len() as f64;
                }
            }else{
                continue;
            }
        }else{
            eprintln!("Error: algn_tuple_list is not all forward or all reverse. This should not happen.");
            continue;
        }

        // Just record the score here; the winner set is picked below.
        scored.push((mcc_idx as u32, norm_sum_joint_prob));
    }

    Ok(scored)
}

// ---------------------------------------------------------------------------
// Equivalence test: production scoring vs this frozen reference, bit-for-bit.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::reference_score_candidates;
    use crate::forseti::{forseti_score_candidates, select_best_mcc_indices, ForsetiCheckingList};
    use crate::mlp_spline::{load_mlp_params_from_str, load_spline_lookup_table_from_str, NativeMlp};
    use crate::seqstore::MemSeqStore;
    use crate::track::TrackStore;
    use std::collections::HashMap;
    use std::sync::Arc;

    const PARAMS: &str = include_str!("../resources/mlp_params_Transpose.json");
    const SPLINE: &str = include_str!("../resources/spline_lookup_table.json");

    /// Tiny deterministic PRNG (xorshift64*), so a failure is reproducible from
    /// its seed without pulling `rand` into the test.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
        fn range(&mut self, lo: usize, hi: usize) -> usize {
            // inclusive range
            lo + self.below((hi - lo + 1) as u64) as usize
        }
        fn chance(&mut self, p_percent: u64) -> bool {
            self.below(100) < p_percent
        }
    }

    /// Transcript-like sequence: uniform ACGT stretches interleaved with A-runs
    /// of assorted lengths (some interrupted by one non-A base, so the 6A scan
    /// sees near-misses), a few Ns, and occasionally a fully-A tail.
    fn gen_transcript(rng: &mut Rng, target_len: usize) -> Vec<u8> {
        const ACGT: [u8; 4] = [b'A', b'C', b'G', b'T'];
        let mut s = Vec::with_capacity(target_len + 64);
        while s.len() < target_len {
            match rng.below(10) {
                0..=4 => {
                    let n = rng.range(10, 150);
                    for _ in 0..n {
                        s.push(ACGT[rng.below(4) as usize]);
                    }
                }
                5..=8 => {
                    let n = rng.range(3, 25);
                    let broken = rng.chance(30);
                    let brk = rng.range(0, n.max(1) - 1);
                    for i in 0..n {
                        if broken && i == brk {
                            s.push(ACGT[1 + rng.below(3) as usize]);
                        } else {
                            s.push(b'A');
                        }
                    }
                }
                _ => {
                    let n = rng.range(1, 3);
                    for _ in 0..n {
                        s.push(b'N');
                    }
                }
            }
        }
        s.truncate(target_len);
        if rng.chance(15) {
            let n = rng.range(1, 40).min(s.len());
            let l = s.len();
            for b in &mut s[l - n..] {
                *b = b'A';
            }
        }
        s
    }

    /// Random splicing status per transcript: 'U' selects the blocked track
    /// storage, anything else the whole-transcript track.
    fn gen_status(rng: &mut Rng, n: usize) -> Vec<u8> {
        (0..n).map(|_| if rng.chance(50) { b'U' } else { b'S' }).collect()
    }

    fn gen_txome(rng: &mut Rng) -> (HashMap<u32, Vec<u8>>, Vec<String>) {
        // lengths chosen to exercise: shorter than a 30-mer, exactly around the
        // tail arm's thresholds, typical spliced, and intron-sized unspliced
        let lens = [
            rng.range(5, 29),
            rng.range(30, 45),
            rng.range(46, 70),
            rng.range(100, 400),
            rng.range(1000, 3000),
            rng.range(3000, 6000),
            rng.range(20_000, 45_000),
        ];
        let mut txome = HashMap::new();
        let mut names = Vec::new();
        for (tid, &l) in lens.iter().enumerate() {
            txome.insert(tid as u32, gen_transcript(rng, l));
            names.push(format!("tx{}", tid));
        }
        (txome, names)
    }

    fn gen_ref_start(rng: &mut Rng, tx_len: usize, read_length: u16, forward: bool) -> u32 {
        if rng.chance(4) {
            // soft-clip sentinel: "close to 2^32" -> treated as position 0
            return u32::MAX - rng.below(read_length as u64 + 45) as u32;
        }
        if forward && rng.chance(2) && tx_len > 0 {
            // beyond the transcript end -> error path, candidate skipped
            return (tx_len + rng.range(1, 100)) as u32;
        }
        let hi = if forward { tx_len.saturating_sub(1) } else { tx_len };
        match rng.below(10) {
            0..=4 => {
                // near the 3' end: tail arm territory
                let lo = tx_len.saturating_sub(1200);
                rng.range(lo, hi) as u32
            }
            5 => rng.range(0, hi.min(40)) as u32,
            _ => rng.range(0, hi) as u32,
        }
    }

    fn gen_checking_list(
        rng: &mut Rng,
        txome: &HashMap<u32, Vec<u8>>,
        read_length: u16,
    ) -> ForsetiCheckingList {
        let n_cand = rng.range(1, 8);
        let n_tx = txome.len() as u64;
        (0..n_cand)
            .map(|_| {
                let tid = rng.below(n_tx) as u32;
                let tx_len = txome[&tid].len();
                let forward = rng.chance(55);
                let mixed = rng.chance(2);
                let n_alns = rng.range(1, 6);
                let alns = (0..n_alns)
                    .map(|i| {
                        let f = if mixed { i % 2 == 0 } else { forward };
                        (f, gen_ref_start(rng, tx_len, read_length, f))
                    })
                    .collect();
                (tid, alns)
            })
            .collect()
    }

    #[test]
    fn production_scoring_matches_frozen_reference() {
        let mlp = Arc::new(NativeMlp::from_params(&load_mlp_params_from_str(PARAMS).unwrap()).unwrap());
        let spline = load_spline_lookup_table_from_str(SPLINE).unwrap();
        let iters: usize = std::env::var("FORSETI_EQ_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30_000);
        let seed: u64 = std::env::var("FORSETI_EQ_SEED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0x9E3779B97F4A7C15);
        let mut rng = Rng(seed);

        let mut n_scored = 0usize;
        let mut n_finite = 0usize;
        let mut n_winners = 0usize;
        let (mut txome, mut names) = gen_txome(&mut rng);
        let mut tracks = TrackStore::new(txome.len(), Arc::new(gen_status(&mut rng, txome.len())), mlp.clone(), Arc::new(MemSeqStore(txome.clone())));
        for it in 0..iters {
            if it % 40 == 0 {
                let t = gen_txome(&mut rng);
                txome = t.0;
                names = t.1;
                tracks = TrackStore::new(txome.len(), Arc::new(gen_status(&mut rng, txome.len())), mlp.clone(), Arc::new(MemSeqStore(txome.clone())));
            }
            let read_length: u16 = if rng.chance(80) { 91 } else { 150 };
            let max_frag_len: u16 = match rng.below(3) {
                0 => 200,
                1 => 1000,
                _ => 1010, // spline table holds 1011 entries
            };
            let list = gen_checking_list(&mut rng, &txome, read_length);

            let got = forseti_score_candidates(
                &list, &names, &spline, &tracks, read_length, max_frag_len,
            );
            let want = reference_score_candidates(
                &list, &names, &txome, &spline, &mlp, read_length, max_frag_len,
            )
            .unwrap();

            let ctx = || format!("iter {} seed {:#x} rl {} mfl {} list {:?}", it, seed, read_length, max_frag_len, list);
            assert_eq!(got.len(), want.len(), "number of scored candidates differs: {}", ctx());
            for (k, (&(gi, gs), &(wi, ws))) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(gi, wi, "mcc_idx differs at {}: {}", k, ctx());
                assert_eq!(
                    gs.to_bits(),
                    ws.to_bits(),
                    "score differs at {} (mcc {}): got {:e} want {:e}: {}",
                    k, gi, gs, ws, ctx()
                );
                n_finite += (gs != f64::NEG_INFINITY) as usize;
            }
            n_scored += got.len();
            let gw = select_best_mcc_indices(&got, 0.0);
            let ww = select_best_mcc_indices(&want, 0.0);
            assert_eq!(gw, ww, "winner set differs: {}", ctx());
            n_winners += gw.len();
        }
        eprintln!(
            "forseti equivalence: {} lists, {} scored candidates ({} finite), {} winners, seed {:#x}",
            iters, n_scored, n_finite, n_winners, seed
        );
        assert!(n_finite * 2 > iters, "generator produced too few finite scores ({})", n_finite);
    }
}
