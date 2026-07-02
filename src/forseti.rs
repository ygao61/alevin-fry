use crate::mlp_spline::predict_with_tch;
use anyhow::{Context, Result, bail};
use ndarray::prelude::*;
use ndarray::{Array1, Array2};
use ndarray::s;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use tch::nn;
type MCCIndex = usize;
type CoveringTxpId = u32;
/// RAD alignment tuple: (is_reverse, ref_start)
type AlgnTuple = (bool, u32);
type ForsetiCheckingList = HashMap<(MCCIndex, CoveringTxpId), Vec<AlgnTuple>>;
use memchr::memmem;


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

#[allow(dead_code)]
fn build_kmers(sequence: &str, ksize: usize) -> Vec<String> {
    // If the sequence is shorter than k, there are no valid k-mers.
    // IMPORTANT: using saturating_sub here is *not* sufficient because it would yield 1
    // and then attempt to slice `sequence[0..ksize]`, which panics.
    if sequence.len() < ksize {
        return Vec::new();
    }
    let n_kmers = sequence.len() - ksize + 1;
    (0..n_kmers)
        .map(|i| sequence[i..i + ksize].to_string())
        .collect()
}

// fn __work_one_hot_encoder(kmer_list: &[String], has_enough_a: &[bool]) -> Array2<f32> {
//     let nucleotides = ['A', 'C', 'G', 'T', 'N'];
//     let num_classes = nucleotides.len();

//     // Create a mapping from ASCII codes to indices
//     let mut code_to_idx = [-1i32; 256];
//     for (i, &nuc) in nucleotides.iter().enumerate() {
//         code_to_idx[nuc as usize] = i as i32;
//     }

//     // Filter valid kmers
//     let valid_kmers: Vec<&String> = kmer_list
//         .iter()
//         .zip(has_enough_a)
//         .filter_map(|(kmer, &has_a)| if has_a { Some(kmer) } else { None })
//         .collect();

//     if valid_kmers.is_empty() {
//         return Array2::<f32>::zeros((0, 0));
//     }

//     let num_sequences = valid_kmers.len();
//     let sequence_length = valid_kmers[0].len();

//     // Prepare the output array
//     let mut one_hot = Array2::<f32>::zeros((num_sequences, sequence_length * num_classes));

//     for (i, seq) in valid_kmers.iter().enumerate() {
//         let seq_bytes = seq.as_bytes();
//         for (j, &byte) in seq_bytes.iter().enumerate() {
//             let idx = code_to_idx[byte as usize];
//             let idx = idx as usize;
//             one_hot[(i, j * num_classes + idx)] = 1.0;
//         }
//     }

//     one_hot
// }
#[allow(dead_code)]
fn one_hot_encoder(kmer_list: &[String], has_enough_a: &Array1<bool>) -> Array2<f32> {
    let nucleotides = ['A', 'C', 'G', 'T', 'N'];
    let num_classes = nucleotides.len();

    // Create a mapping from ASCII codes to indices
    let mut code_to_idx = [-1i32; 256];
    for (i, &nuc) in nucleotides.iter().enumerate() {
        code_to_idx[nuc as usize] = i as i32;
    }

    // Filter valid kmers using ndarray boolean masking
    let valid_kmers: Vec<&String> = kmer_list
        .iter()
        .zip(has_enough_a.iter())
        .filter_map(|(kmer, &has_a)| if has_a { Some(kmer) } else { None })
        .collect();

    if valid_kmers.is_empty() {
        return Array2::<f32>::zeros((0, 0));
    }

    let num_sequences = valid_kmers.len();
    let sequence_length = valid_kmers[0].len();

    // Prepare the output array
    let mut one_hot = Array2::<f32>::zeros((num_sequences, sequence_length * num_classes));

    // Process each kmer
    for (i, seq) in valid_kmers.iter().enumerate() {
        let seq_bytes = seq.as_bytes();
        let indices: Vec<usize> = seq_bytes
            .iter()
            .map(|&byte| code_to_idx[byte as usize] as usize)
            .collect();

        // Vectorized one-hot encoding for the current sequence
        for (j, &idx) in indices.iter().enumerate() {
            one_hot[(i, j * num_classes + idx)] = 1.0;
        }
    }

    one_hot
}

fn load_spline_lookup_table(file_path: &str) -> Result<Array1<f64>> {
    // Open the file
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    // Parse the JSON
    let json_data: Value = serde_json::from_reader(reader)?;

    // Extract the "y" field as an array
    if let Some(y_values) = json_data["y"].as_array() {
        // Convert JSON array to Vec<f64>
        let y_vec: Vec<f64> = y_values
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0)) // Ensure conversion
            .collect();

        // Convert Vec<f64> to ndarray::Array1
        Ok(Array1::from(y_vec))
    } else {
        Err(anyhow::anyhow!("Missing 'y' field in JSON"))
    }
}
/// Allocation-free 6A detection. For each k-mer window start `i` in `[0, len-k]`,
/// returns whether `bytes[i..i+k]` contains a run of >= `min_run` consecutive b'A'.
/// O(len) via a difference array; bit-equivalent to the old per-k-mer
/// `memmem::find(window, "A"*min_run).is_some()` but with zero String allocations.
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

/// One-hot encode the k-mers starting at `indices`, reading directly from `bytes`
/// (no per-k-mer String allocation). Same A,C,G,T,N -> 0..4 column layout as the
/// previous `one_hot_encoder`.
fn one_hot_from_bytes(bytes: &[u8], k: usize, indices: &[usize]) -> Array2<f32> {
    let num_classes = 5usize; // A, C, G, T, N
    let mut code_to_idx = [-1i32; 256];
    for (i, &nuc) in [b'A', b'C', b'G', b'T', b'N'].iter().enumerate() {
        code_to_idx[nuc as usize] = i as i32;
    }
    if indices.is_empty() {
        return Array2::<f32>::zeros((0, 0));
    }
    let mut one_hot = Array2::<f32>::zeros((indices.len(), k * num_classes));
    for (row, &start) in indices.iter().enumerate() {
        for j in 0..k {
            let idx = code_to_idx[bytes[start + j] as usize] as usize;
            one_hot[(row, j * num_classes + idx)] = 1.0;
        }
    }
    one_hot
}

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

// Global hit/miss counters for the per-thread MLP affinity cache. One atomic add
// per process_binding_affinity call (not per 30-mer) -> negligible contention.
// Read at end of run via `mlp_cache_stats()` to report the hit rate.
pub static MLP_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
pub static MLP_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

/// (hits, misses) for the MLP affinity cache so far.
pub fn mlp_cache_stats() -> (u64, u64) {
    (
        MLP_CACHE_HITS.load(Ordering::Relaxed),
        MLP_CACHE_MISSES.load(Ordering::Relaxed),
    )
}

thread_local! {
    // packed 30-mer -> FINAL affinity (post discount / all-A / threshold). Per-thread
    // => no locks, scales with -t. Recurrent multimap loci (Hmgb2 cluster, ribosomal
    // pseudogenes) appear in every cell, so each worker's cache fills within a few cells.
    static MLP_AFFINITY_CACHE: RefCell<HashMap<u64, f64>> = RefCell::new(HashMap::new());
}

/// Pack a k-mer (k <= 32) of A/C/G/T into a u64 (2 bits/base). None if any base is
/// not A/C/G/T (e.g. N) -> those are computed each time but never cached.
#[inline]
fn pack_kmer(bytes: &[u8], start: usize, k: usize) -> Option<u64> {
    let mut key = 0u64;
    for j in 0..k {
        let code = match bytes[start + j] {
            b'A' => 0u64,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => return None,
        };
        key = (key << 2) | code;
    }
    Some(key)
}

fn process_binding_affinity(
    bytes: &[u8],
    k: usize,
    has_enough_a: &Array1<bool>,
    mlp: &nn::Sequential,
    discount_perc: f64,
    binding_affinity_threshold: f64,
) -> Result<Array1<f64>> {
    // hot window-start indices
    let indices: Vec<usize> = has_enough_a
        .indexed_iter()
        .filter_map(|(i, &val)| if val { Some(i) } else { None })
        .collect();

    let mut out = Array1::<f64>::zeros(has_enough_a.len());

    // 1) split hot 30-mers into cache HITS (fill `out` directly) and MISSES (need MLP)
    let mut miss_start: Vec<usize> = Vec::new();
    let mut miss_key: Vec<Option<u64>> = Vec::new();
    let mut hits: u64 = 0;
    MLP_AFFINITY_CACHE.with(|c| {
        let cache = c.borrow();
        for &start in &indices {
            let key = pack_kmer(bytes, start, k);
            if let Some(kk) = key {
                if let Some(&aff) = cache.get(&kk) {
                    out[start] = aff;
                    hits += 1;
                    continue;
                }
            }
            miss_start.push(start);
            miss_key.push(key);
        }
    });

    // 2) one batched MLP forward on the MISSES only, then post-process + cache
    if !miss_start.is_empty() {
        let encoded = one_hot_from_bytes(bytes, k, &miss_start);
        let mut nonzero = predict_with_tch(&mlp, encoded)?;
        if nonzero.len() != miss_start.len() {
            return Err(anyhow::anyhow!(
                "Mismatched dimensions between binding affinity and has_enough_a"
            ));
        }
        nonzero *= discount_perc;
        MLP_AFFINITY_CACHE.with(|c| {
            let mut cache = c.borrow_mut();
            for (j, &start) in miss_start.iter().enumerate() {
                // identical post-processing as before: all-A -> 1.0; <= threshold -> 0.0
                let aff = if bytes[start..start + k].iter().all(|&b| b == b'A') {
                    1.0
                } else if nonzero[j] <= binding_affinity_threshold {
                    0.0
                } else {
                    nonzero[j]
                };
                out[start] = aff;
                if let Some(kk) = miss_key[j] {
                    cache.insert(kk, aff);
                }
            }
        });
    }

    // 3) record hit/miss (one atomic add each per call)
    MLP_CACHE_HITS.fetch_add(hits, Ordering::Relaxed);
    MLP_CACHE_MISSES.fetch_add(miss_start.len() as u64, Ordering::Relaxed);

    Ok(out)
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

pub fn forseti_for_multi_best(
    forseti_checking_list: &ForsetiCheckingList,
    ref_names: &[String],
    spliceu_txome: &HashMap<u32, Vec<u8>>,
    spline_lookup: &Array1<f64>,
    mlp: &nn::Sequential,
    read_length: u16,
    max_frag_len: u16,
) -> Result<Vec<u16>> {
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
    let mut best_mcc_indices: Vec<u16> = Vec::new();
    let mut max_score = f64::NEG_INFINITY;


    // algn_tuple_list is direction(fw, reverse) and ref_start
    for ((mcc_idx, covering_txp_id), algn_tuple_list) in forseti_checking_list {
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
                    let downstream_binding_affinity = process_binding_affinity(
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
            if overlap_wdow_end > tx_ref_end - 30 {
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
                    process_binding_affinity(
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
                    let upstream_binding_affinity = process_binding_affinity(
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

        // Update the best scores and indices in a single pass
        if norm_sum_joint_prob > max_score {
            max_score = norm_sum_joint_prob;
            best_mcc_indices.clear();
            best_mcc_indices.push(*mcc_idx as u16);
        } else if (norm_sum_joint_prob - max_score).abs() < 1e-6 && norm_sum_joint_prob != f64::NEG_INFINITY {
            best_mcc_indices.push(*mcc_idx as u16);
        }
    }

    if best_mcc_indices.is_empty() {
        return Ok(Vec::new());
    }


    Ok(best_mcc_indices)
}
