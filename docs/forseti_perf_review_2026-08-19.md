# alevin-fry `forseti` branch — memory & speed review (2026-08-19)

Scope: the full Forseti path — `generate-permit-list` → `collate` → `quant -r forseti-parsimony-em`
(`src/cellfilter.rs`, `src/collate.rs`, `src/quant.rs::do_quantify_forseti`, `src/eq_class.rs`,
`src/pugutils.rs::get_num_molecules_forseti`, `src/forseti.rs`, `src/mlp_spline.rs`, plus libradicl
`7c65a1e` for the position-carrying record). HEAD = `a9b2514`.

## 0. Measured reference point (Allen MTG TX0029-12, 452 M reads, 227,436 chunks, `-t 32`)

| step | wall | CPU-s | max RSS |
|---|---|---|---|
| quant parsimony-em | 1:19 | 1,666 | 35.2 GB |
| quant forseti-parsimony-em | **17:40** | **32,585** | **42.2 GB** |

`af_quant.log`: `Forseti MLP cache: 121,709,858,014 hits + 429,336,076 misses (99.6 %)`.
spliceu FASTA (`human_GENCODE_FILTERED_v49_spliceu/ref/roers_ref.fa`): **23.5 GB, 976,514 records**,
single-line sequences, `.fai` present. MLP = 150→100→1 (relu, logistic); spline table = 1,011 f64.

Forseti quant costs ~20× the CPU of the baseline quant. The 122 billion cache lookups are the dominant
cost (≈100–250 ns each incl. `pack_kmer` + SipHash probe ≈ 12–30 k CPU-s of the 32.6 k).

---

## 1. Correctness issues found along the way (fix before optimizing)

### 1.1 HIGH — UMI index mismatch between `eqc_info` and `eqc_info_forseti`
`eq_class.rs:569-693` (`init_from_chunk_forseti`). `eqc_info[eq].umis` is sorted + collapsed at the end
(lines 675-691) and the PUG vertices are `(eqid, xi)` with `xi` indexing that **sorted** list
(`pugutils.rs:141-148`). But `eqc_info_forseti[eq].umis` is left in **first-seen order**
(lines 625-631, 651-654) and is indexed with the same `umi_idx` in `get_check_mcc_list` /
`get_check_mcc_list_from_map` (`pugutils.rs:1864-1866, 1891-1893`). For any eq-class whose UMIs did not
first appear in ascending order (i.e. almost every class with ≥ 3 UMIs), Forseti scores an MCC with the
alignment positions of a *different* molecule of the same eq-class. Reads in one eq-class sit in the
same transcript set and often near each other, which is why results still look sensible — but it is
wrong and is a free accuracy gain. Fix: build one structure after sorting reads by `(eq, umi)`:
`umis: Vec<(u64, u32 count)>`, `read_idx: Vec<u32>` flat + `read_start: Vec<u32>`; drop
`eqc_info_forseti`. (Also removes the O(U²) `find` below.)

### 1.2 HIGH — small-cell branch zero-fills Forseti cells
`quant.rs:2047-2079`: cells with `< small_thresh` (=10) reads take the `else` branch; `ForsetiParsimonyEm`
falls to `_ =>` → `counts = vec![0; num_genes]` (wrong length, should be `num_rows`) + one `warn!` per
cell. Those cells are written as all-zero rows, land in `empty_resolved_cells`, and give NaN
`MeanByMax`. Fix: add `ForsetiParsimonyEm` to the `CellRangerLikeEm | ParsimonyEm | ParsimonyGeneEm`
arm (uniform split) and make the fallback `vec![0; num_rows]`.

### 1.3 HIGH — latent panic in collate thread-local bucket buffer (forseti-exposed) — **fixed (fix16)**
`collate.rs:492-498`: `loc_buffer_size` lower bound assumes 4 B/alignment (`24 + most_ambig*4`), but the
position record is `20 + 8*na` bytes (libradicl `record.rs:324-340`). With many threads/buckets the
clamp term shrinks (e.g. 64 threads, `-m 30 M` → ≈ 10.9 KB) and a read with > ~1,360 retained
alignments makes `rr.write(bcursor)` fail → `.expect("can write record")` abort. Fix:
`loc_buffer_size = R::nbytes(most_ambig, ctx).max(clamp(...))`.

### 1.4 MED — `--spliceu-fa` is `required(true)` for *every* `quant` (`main.rs:157`), no `usa_mode` /
`rlen` guard for Forseti (`quant.rs:1479-1485, 1974-2023`): on a 2-column t2g `extract_usa_eqmap` runs
with `usa_offsets=None` → garbage indices. Fix: optional arg; `bail!` if
`resolution==ForsetiParsimonyEm && (!usa_mode || spliceu_fa.is_none() || rlen missing)`.

### 1.5 LOW — numerical/robustness
- Spline table has 29 negative entries (idx 0-6, 37-58, ≈ −1e-5). `(affinity*frag_prob + 1e-12).ln()`
  is NaN there; NaN positions are silently dropped by the `f64::max` fold. Clamp `y.max(0.0)` at load.
- `forseti_checking_list` is a `std::HashMap` (random `RandomState`) iterated to pick winners
  (`forseti.rs:410`); tie order is therefore per-process random. It is keyed by `(mcc_idx, txp)` which
  is just the index into `check_mcc_list` → make it a `Vec<(u32 txp, Vec<AlgnTuple>)>`.
  *Update 2026-08-23 (two independent code reviews):* the winner **set** is order-independent since
  `3b0ad4a`, and the per-candidate hash is < 1 % of candidate cost (the MLP cache lookups dominate), so
  this is hygiene, not speed. Order still leaks through (i) `filtered_mcc_txp_pairs.iter().next()`
  (`pugutils.rs:1675`) on cross-MCC ties and (ii) which 30-mers are batched into one libtorch call —
  so a Vec gives a *different* deterministic order, not the current one; verify run-to-run identity
  empirically. Cheapest determinism: `ahash::RandomState::with_seeds(2,7,1,8)` like the rest of the
  crate. `INVALID_POS_PRINT_LIMIT`/`invalid_pos_skipped` (`forseti.rs:392-393`) are dead code.
  **Done (fix17):** `Vec<(txp, Vec<AlgnTuple>)>` + explicit tie-break (MCC of the lowest-index winner).
  pbmc_1k_v3 on one collated RAD: fix16 vs fix16 moved 4 UMIs between genes (random tie picks);
  fix17 run1 vs run2 moved 0, residual ≤ 1e-5 on EM-split entries (parsimony-EM float order, not forseti).
- `spliceu` FASTA load: `record.id()` is the full header; no check `spliceu_txome.len()==ref_count`
  (`quant.rs:1533-1538`). Split at whitespace and assert.
- ~~`--max-frag-len` (default 1000) is not validated against the spline length (1011); > 1010 panics on
  slice (`forseti.rs:499`).~~ **Fixed (fix16):** `quant.rs` bails after the spline loads; `main.rs` now
  returns the error chain instead of `panic!("could not quantify rad file.")`, which used to hide it.

---

## 2. Speed

### 2.1 HIGH — re-scoring the same transcript positions 122 × 10⁹ times — **done (fix19, 2026-08-24)**
Implemented in `src/track.rs` (`TrackStore`): per-transcript hot-position tracks (spliced: whole transcript;
unspliced: lazily built 1024-bp blocks, chosen from a 4096/1024/512 sweep), process-wide `OnceLock` slots,
`f32` affinities, tail windows precomputed. The three scoring arms in `forseti.rs` are range scans over hot
positions; control flow and summation order are unchanged. Equivalence is proven three ways: the frozen fix18
copy in `src/forseti_reference.rs` + randomized bit-exact test (`cargo test --release`), the
`--features forseti-shadow` build that re-scores every real candidate list with the reference (pbmc_1k:
8.2 M lists / 221.7 M candidates; pbmc_10k: 82.6 M lists / 2.29 G candidates; 0 mismatches), and `collated_rad/compare.py` vs the fix18 reference quants
(indistinguishable from a fix18 rerun). Measured on EPYC-7313, 32 threads, warm RAD:
pbmc_10k forseti quant 16:13 → 6:39 (2.4×), RSS 44.7 → 47.9 GB (+7%); pbmc_1k 1:50 → 1:12.
Track building is ~1000 CPU-s (30 s wall) and the per-thread MLP cache (3.2) is gone. Reuse analysis (per-block
query counts, `FORSETI_TRACK_REPORT=<tsv>`): S tracks < 200 MB with 95–99.6 % of queries on blocks queried ≥100×;
U blocks carry all the memory, and blocks queried ≥10× serve ~99 % of queries; the 1–2× tier is ≤ 1.7 GB
(3.6 % of RSS) so no eviction policy was added. Original analysis follows.

Binding affinity is a property of a *transcript position*, yet it is recomputed (via a per-thread 30-mer
hash cache) for every (MCC, transcript) candidate in every cell: `pack_kmer` (30-iteration loop) +
SipHash lookup per hot 30-mer (`forseti.rs:239-288`), and the log-sum loops run over **all** `n_kmers`
positions × `n_alns` (`forseti.rs:495-512, 564-588, 666-682`) although cold positions contribute a
constant `ln_eps` and can never beat a hot position.

Recommended redesign (biggest win, probably 5–20× on quant):
1. Per-transcript, lazily computed, **shared** "hot-position track": `Vec<(u32 pos, f32 aff)>` sorted by
   position (or dense `u8`-quantized affinity for touched transcripts), stored in a
   `Vec<OnceLock<Arc<…>>>` indexed by tid (or `DashMap<u32, Arc<…>>` — `dashmap` is already a
   dependency). Compute `compute_has_6a` + MLP once per transcript (per process, not per thread).
2. Score a window as a range scan over hot positions only: `O(#hot × n_alns)` instead of
   `O(n_kmers × n_alns)` + `O(n_kmers)` hashing. Equivalent result (cold positions ≥ `ln_eps` only).
3. Tail (`format!("{}A…")`, `forseti.rs:525-529`) and reverse-complement (`reverse_complement` →
   `String`, `forseti.rs:645`) windows: handle with a virtual poly-A extension / a byte LUT into a
   reusable scratch buffer; no per-candidate `String`.
Cheap interim steps if the redesign waits: rolling 2-bit key instead of `pack_kmer` (O(1)/position);
`ahash`/`FxHash` instead of SipHash for `MLP_AFFINITY_CACHE` (≈2× on lookups); store `f32`; make the
cache process-global (`DashMap`) so each 30-mer is computed once, not once per thread.

### 2.2 HIGH — libtorch for a 150→100→1 MLP
`mlp_spline.rs:63-137`: per call: `Tensor::from_slice` + reshape + dispatcher + `Vec<f64>` conversion
(≈ 10–50 µs overhead for a 15 k-MAC network). No `tch::set_num_threads(1)` anywhere → each of the 31
workers owns a model whose intra-op pool defaults to all cores (involuntary ctx-switches 331 k vs 20 k
baseline). Fix now: `tch::set_num_threads(1); tch::set_num_interop_threads(1);` at the top of
`do_quantify_forseti`. Fix properly: hand-written f32 `matvec` (two `for` loops, or `ndarray::dot`) —
removes the libtorch dependency (LIBTORCH env, ~0.5–1 GB of shared libs, build pain) entirely.

### 2.3 HIGH — O(U²) UMI lookup in `init_from_chunk_forseti`
`eq_class.rs:625`: `fe.umis.iter_mut().find(|(u,_)| *u == r.umi())` per read → quadratic in the number
of distinct UMIs of an eq-class. A nucleus with MALAT1-U ≈ 50 k UMIs costs ~1.2 × 10⁹ comparisons for
that one cell. Fixed for free by the sort-based rebuild in 1.1.

### 2.4 MED — allocations sized by the *whole cell graph* inside the MCC search
`pugutils.rs:430, 438` (`collapse_vertices_keep_ties`): `HashMap::with_capacity(g.node_count())` per call
and `visited_set` with capacity `g.node_count()` per transcript, called for every uncovered vertex in
every `while` iteration (`pugutils.rs:1501-1543`). For a 10⁵–10⁶-node cell that is MBs of allocation per
BFS, O(V²) traffic. (`visited_set` sizing is inherited from master; the map is new.) Fix: size by
`comp_verts.len()`, keep one reusable visited bitmap (component-local indices) and one reusable map
across calls; return results into caller-owned buffers.

### 2.5 MED — per-candidate allocation churn in `forseti_for_multi_best`
Per (MCC, txp): `compute_has_6a` (2 Vecs), `process_binding_affinity` (≥ 4 Vecs + `Array2` one-hot),
`Array1` outputs, `tail_seq` String, `reverse_complement` String, `check_mcc_list` clones
(`pugutils.rs:1905-1908`). ~10 allocs × ~10⁹ candidates. Put all scratch in a per-thread
`ForsetiScratch` struct and pass `&mut`.

### 2.6 MED — collate / generate-permit-list (libradicl + collate.rs)
- 7 heap allocs per record in the collate scatter (`lib.rs:876-885` → `record.rs:1476-1518`: dirs/refs/pos
  + argsort + 3× `visited`); master did 2. Parse into one `SmallVec<[(u32,u32,bool);16]>` and sort once,
  or patch the barcode in the raw 8-byte slots without materialising `R`.
- `generate-permit-list` fully parses every record (3 Vecs) but only needs `bc`, `na`, strand bit
  (`cellfilter.rs:987-1001`). Add a non-allocating header+strand scan.
- Busy-spin loops on `ArrayQueue` (`collate.rs:545-559, 687-711`, `readers.rs:259-262`,
  `cellfilter.rs:817-831`) burn all cores while the single reader is the bottleneck → `crossbeam_channel::
  bounded` or `spin_loop()/yield_now()` backoff.
- Two SipHash lookups + two `SeqCst fetch_add` on shared atomics per record in scatter (`lib.rs:879-930`)
  → one `ahash` map `raw_bc → (corrected_bc, bucket)`, per-thread counters flushed per bucket.
- Unfiltered mode materialises one `u64` per unmatched read and sorts them (`cellfilter.rs:274-318, 800`).
- `collate.rs:889-1472` is ~580 lines of commented-out old code; `tsv_map.clone()` ×3 (`:852,865,877`).

### 2.7 LOW — quant loop
- `em_optimize_subset` allocates `alphas_in/out` (2 × num_rows) per cell (`em.rs:189-190`);
  `fill_ref_offsets` allocates an nref Vec per cell (`eq_class.rs:285-294`); `EqMap::clear` memsets
  976 k `label_counts` per cell. Keep buffers in the worker.
- Writer mutex holds gzip + featureDump formatting (`quant.rs:2204-2263`); pre-format outside the lock.
- `get_forseti_check_list` rescans `r.refs()` per (read, txp) — fine for small `na`, but positions could be
  carried on the eq-class entry directly once 1.1 is done.

---

## 3. Memory

### 3.1 HIGH — 23.5 GB spliceu transcriptome as `HashMap<u32, Vec<u8>>` (`quant.rs:1521-1538`) — **done (tag forseti-fix20)**
Implemented as `src/seqstore.rs`: the tracks (2.1) read a transcript's bytes exactly once while a block/tail is
built, so the FASTA is no longer parsed or held; `FaiSeqStore` keeps the `.fai` index and serves each request with
a positional read from the (shared, evictable) page cache. Measured (EPYC-7313, 32 threads, FASTA + `.fai` staged
to local disk): pbmc_10k forseti quant RSS 47.7 → 17.5 GB, wall 6:38 → 6:30 (no start-up parse); pbmc_1k RSS
39.7 → 11.1 GB, wall 1:12 → 1:05. Reads: 14.2 M fetches / 14.8 GB on pbmc_10k. Scores unchanged (synthetic
test, shadow on pbmc_1k 0 mismatches, counts within the same-binary control). Requires `<fasta>.fai`
(`samtools faidx`); a cold network FS is slow for the random reads — stage the FASTA locally. Original analysis:

976,514 separate `Vec<u8>` + hashmap; ~23.5 GB resident for the whole run, parsed single-threaded before
any worker starts. Options, from smallest change to best:
- `Vec<Vec<u8>>` / flat `Vec<u8>` + offsets indexed by tid (removes hashing + 1 M allocations; same RSS).
- **2-bit pack** (N → A with a side bitmap, or 3 bits): ≈ 6 GB; windows decoded into scratch on use.
- **mmap the FASTA via the `.fai`** (sequences are single-line; `offset,len` per record): zero parse time,
  RSS = touched pages only (Forseti touches only windows around multi-mapping reads, likely ≪ 5 GB),
  pages shared by all threads. Caveat: random 1 KB reads from a network FS (`/fs/nexus-projects`) are
  slow if the file is not in page cache — stage the FASTA to local scratch or rely on the node cache.
- Structural: unspliced isoforms of a gene overlap almost completely; a per-gene genomic span + per-transcript
  (gene, offset) would cut the unspliced part several-fold (needs a roers change).
Also: only load when `resolution == ForsetiParsimonyEm` (currently unconditional on the RnaShortPos path).

### 3.2 HIGH — per-thread, unbounded MLP cache ≈ 7–9 GB — **removed by fix19 (replaced by the track store, see 2.1)**
`forseti.rs:229-234`: `thread_local! HashMap<u64, f64>`; 429 M misses total → 429 M entries across 31
threads × ~20 B (key + f64 + hashbrown overhead) ≈ 8.6 GB, and the same 30-mer is recomputed once per
thread. This is most of the 42 − 35 = 7 GB delta over baseline quant. Fix: process-global shared cache
(`DashMap<u64, f32>`, sharded) or — better — the per-transcript hot-position track of 2.1, which bounds
memory by touched transcripts and removes the hash entirely.

### 3.3 MED — position record = 3 `Vec`s per read (libradicl `record.rs:429-435`)
`dirs: Vec<bool>`, `refs: Vec<u32>`, `pos: Vec<u32>` per record (master: 1 Vec). Use one
`SmallVec<[(u32 ref, u32 pos, bool); 8]>` or a single flat `Vec<u32>` with refs/pos interleaved; `dirs`
can be a `u64` bitmask for `na ≤ 64`.

### 3.4 MED — in-memory triplet matrix (`quant.rs:1623-1627, 1656`) [pre-existing]
`tmcap = 0.1·num_genes·num_cells` reserved with `num_genes` (not `num_rows`, so USA mode under-reserves
and reallocates three large Vecs under the writer lock). For the 227 k-chunk unfiltered runs this is
multi-GB; `--use-eds` streams instead.

### 3.5 MED — libtorch itself (≈ 0.5–1 GB of shared libraries, OMP pools per worker) → see 2.2.

### 3.6 OPEN — baseline quant already sits at 33–35 GB on the 227 k-chunk RADs
That is not explained by anything I can see in `do_quantify` (metachunk queue is bounded at
4·n_workers × ~512 KB; per-thread structures are small). Candidates: libradicl `fill_work_queue` never
shrinks `buf` after one huge cell and `buf.clone()`s the full buffer for every later metachunk
(`readers.rs:199-255`); the triplet matrix; mimalloc retention. Worth one `heaptrack`/`massif` run of
`quant -r parsimony-em` on TX0029-12 before optimizing anything else on the memory side — the forseti
delta (+7 GB) is smaller than the spliceu copy alone (23 GB), which means the baseline peak is
data-proportional and shared.

---

## 4. Portability / hygiene
- Hard-coded absolute paths for `mlp_params_Transpose.json` and `spline_lookup_table.json`
  (`quant.rs:1513, 1517`; 315 KB + 23 KB). `include_str!` them into the binary (or CLI flags).
- `do_quantify_forseti` is a ~1,100-line copy of `do_quantify` differing in one match arm + resource
  loading; merge back (a closure/trait for the per-cell resolver) so fixes don't diverge.
- Dead code: `build_kmers`, `one_hot_encoder`, duplicate `load_spline_lookup_table` (forseti.rs and
  mlp_spline.rs), commented `collate_with_temp` copy.
- `eprintln!`/`println!` inside hot loops (`forseti.rs:446-447, 690`) — route through the logger with a
  counter cap.
- Allow `keep-dir`-less RADs to fail fast with a clear message.

---

## 5. Suggested order of work
1. Fix 1.1 (UMI index mismatch) + 1.2 (small cells) + 1.3 (collate buffer) — correctness; re-run one
   sample and compare matrices (expect small but real changes from 1.1).
2. `tch::set_num_threads(1)` + `ahash` cache + f32 values + `Vec` instead of `HashMap` for
   `forseti_checking_list` — one afternoon, measurable.
3. Native MLP (drop libtorch) + `include_str!` params.
4. ~~Per-transcript hot-position tracks + hot-only scoring (2.1)~~ **done (fix19)**; replaced 3.2.
5. ~~spliceu storage (3.1)~~ **done (forseti-fix20)**: `.fai` + positional reads, nothing resident.
6. libradicl record / collate allocation work (2.6, 3.3) — benefits all modes.
7. Profile baseline quant memory (3.6).
