# alevin-fry `forseti` branch — memory & speed review and tracker

Opened 2026-08-19 as a review of HEAD `a9b2514`; since then kept as the running tracker for the Forseti
performance/correctness work. The analysis text of each item is left as written on 2026-08-19; the status
line under each heading, the table below and the update log at the end are what changes.

## Status (updated 2026-08-25)

| item | what | status | when | where |
|---|---|---|---|---|
| 1.1 | UMI index mismatch `eqc_info` vs `eqc_info_forseti` | done | 2026-08-21 | a19b8ca |
| 1.2 | small-cell branch zero-fills Forseti cells | done | 2026-08-21 | a19b8ca |
| 1.3 | collate thread-local bucket buffer panic | done | 2026-08-23 | df228d1 |
| 1.4 | `--spliceu-fa` required for every quant; load only for Forseti | done | 2026-08-21 | a19b8ca |
| 1.5 | candidate order / tie-break non-determinism | done | 2026-08-23 | c0d45fb |
| 2.1 | re-scoring transcript positions per candidate → per-transcript tracks | done | 2026-08-24 | forseti-fix19 |
| 2.2 | libtorch for the MLP → native `NativeMlp` | done | 2026-08-23 | forseti-fix18 |
| 2.3 | O(U²) UMI lookup in `init_from_chunk_forseti` | done (with 1.1) | 2026-08-21 | a19b8ca |
| 2.4 | MCC search allocations sized by the whole cell graph | open | | |
| 2.5 | per-candidate allocation churn in `forseti_for_multi_best` | mostly gone with 2.1 | 2026-08-24 | forseti-fix19 |
| 2.6 | collate / gpl allocations (libradicl) | open | | |
| 2.7 | quant-loop buffer reuse, writer lock | open | | |
| 3.1 | 23.5 GB spliceu in memory → `.fai` on-demand reads | done | 2026-08-24 | forseti-fix20 |
| 3.2 | per-thread unbounded MLP cache | removed by 2.1 | 2026-08-24 | forseti-fix19 |
| 3.3 | position record = 3 `Vec`s per read (libradicl) | open | | |
| 3.4 | triplet-matrix reserve uses `num_genes` | open | | |
| 3.5 | libtorch shared libraries / OMP pools | gone with 2.2 | 2026-08-23 | forseti-fix18 |
| 3.6 | baseline quant 33–35 GB unexplained | resolved by 1.4 (it was the unconditional spliceu load; P now 6–11 GB) | 2026-08-21 | a19b8ca |
| 4 | `include_str!` model files | done | 2026-08-23 | 3b0ad4a |
| 4 | dead code (`build_kmers`, `one_hot_encoder`, dup spline loader) | done | 2026-08-24 | forseti-fix19 |
| 4 | merge `do_quantify_forseti` back into `do_quantify` | open | | |
| 4 | `eprintln!` in hot loops → capped logger | done (capped warning; resolver `exit(1)` sites removed) | 2026-08-25 | 1c68c8c, 2b028e2 |

Measured on pbmc_10k_v3 (486 M reads, EPYC-7313, 32 threads, inputs on local disk), forseti-parsimony-em quant:

| build | wall | peak RSS |
|---|---|---|
| a9b2514 (July benchmark, unpinned libtorch) | 14:50 | 42.3 GB |
| forseti-fix18 | 16:13 | 44.8 GB |
| forseti-fix19 | 6:38 | 47.7 GB |
| forseti-fix20 | 6:30 | 17.5 GB |

Validation stack for every change: frozen per-window scorer + randomized test (`src/forseti_reference.rs`),
`--features forseti-shadow` on real data, `nf_pipeline/collated_rad/` gates (exact pre-EM eq-class comparison,
post-EM `compare.py`). Details and figures: `analysis/fix_per_txp_cache/report.md`.

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

### 1.1 HIGH — UMI index mismatch between `eqc_info` and `eqc_info_forseti` — **done 2026-08-21 (a19b8ca)**
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

### 1.2 HIGH — small-cell branch zero-fills Forseti cells — **done 2026-08-21 (a19b8ca)**
`quant.rs:2047-2079`: cells with `< small_thresh` (=10) reads take the `else` branch; `ForsetiParsimonyEm`
falls to `_ =>` → `counts = vec![0; num_genes]` (wrong length, should be `num_rows`) + one `warn!` per
cell. Those cells are written as all-zero rows, land in `empty_resolved_cells`, and give NaN
`MeanByMax`. Fix: add `ForsetiParsimonyEm` to the `CellRangerLikeEm | ParsimonyEm | ParsimonyGeneEm`
arm (uniform split) and make the fallback `vec![0; num_rows]`.

### 1.3 HIGH — latent panic in collate thread-local bucket buffer (forseti-exposed) — **done 2026-08-23 (df228d1)**
`collate.rs:492-498`: `loc_buffer_size` lower bound assumes 4 B/alignment (`24 + most_ambig*4`), but the
position record is `20 + 8*na` bytes (libradicl `record.rs:324-340`). With many threads/buckets the
clamp term shrinks (e.g. 64 threads, `-m 30 M` → ≈ 10.9 KB) and a read with > ~1,360 retained
alignments makes `rr.write(bcursor)` fail → `.expect("can write record")` abort. Fix:
`loc_buffer_size = R::nbytes(most_ambig, ctx).max(clamp(...))`.

### 1.4 MED — **done 2026-08-21 (a19b8ca)** — `--spliceu-fa` is `required(true)` for *every* `quant` (`main.rs:157`), no `usa_mode` /
`rlen` guard for Forseti (`quant.rs:1479-1485, 1974-2023`): on a 2-column t2g `extract_usa_eqmap` runs
with `usa_offsets=None` → garbage indices. Fix: optional arg; `bail!` if
`resolution==ForsetiParsimonyEm && (!usa_mode || spliceu_fa.is_none() || rlen missing)`.

### 1.5 LOW — numerical/robustness — **done 2026-08-23 (c0d45fb)**
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

### 2.1 HIGH — re-scoring the same transcript positions 122 × 10⁹ times — **done 2026-08-24 (forseti-fix19)**
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

### 2.2 HIGH — libtorch for a 150→100→1 MLP — **done 2026-08-23 (forseti-fix18)**
`mlp_spline.rs:63-137`: per call: `Tensor::from_slice` + reshape + dispatcher + `Vec<f64>` conversion
(≈ 10–50 µs overhead for a 15 k-MAC network). No `tch::set_num_threads(1)` anywhere → each of the 31
workers owns a model whose intra-op pool defaults to all cores (involuntary ctx-switches 331 k vs 20 k
baseline). Fix now: `tch::set_num_threads(1); tch::set_num_interop_threads(1);` at the top of
`do_quantify_forseti`. Fix properly: hand-written f32 `matvec` (two `for` loops, or `ndarray::dot`) —
removes the libtorch dependency (LIBTORCH env, ~0.5–1 GB of shared libs, build pain) entirely.

### 2.3 HIGH — O(U²) UMI lookup in `init_from_chunk_forseti` — **done 2026-08-21 (with 1.1)**
`eq_class.rs:625`: `fe.umis.iter_mut().find(|(u,_)| *u == r.umi())` per read → quadratic in the number
of distinct UMIs of an eq-class. A nucleus with MALAT1-U ≈ 50 k UMIs costs ~1.2 × 10⁹ comparisons for
that one cell. Fixed for free by the sort-based rebuild in 1.1.

### 2.4 MED — allocations sized by the *whole cell graph* inside the MCC search — open
`pugutils.rs:430, 438` (`collapse_vertices_keep_ties`): `HashMap::with_capacity(g.node_count())` per call
and `visited_set` with capacity `g.node_count()` per transcript, called for every uncovered vertex in
every `while` iteration (`pugutils.rs:1501-1543`). For a 10⁵–10⁶-node cell that is MBs of allocation per
BFS, O(V²) traffic. (`visited_set` sizing is inherited from master; the map is new.) Fix: size by
`comp_verts.len()`, keep one reusable visited bitmap (component-local indices) and one reusable map
across calls; return results into caller-owned buffers.

### 2.5 MED — per-candidate allocation churn in `forseti_for_multi_best` — **mostly gone with 2.1 (2026-08-24)**; remaining buffers are reused
Per (MCC, txp): `compute_has_6a` (2 Vecs), `process_binding_affinity` (≥ 4 Vecs + `Array2` one-hot),
`Array1` outputs, `tail_seq` String, `reverse_complement` String, `check_mcc_list` clones
(`pugutils.rs:1905-1908`). ~10 allocs × ~10⁹ candidates. Put all scratch in a per-thread
`ForsetiScratch` struct and pass `&mut`.

### 2.6 MED — collate / generate-permit-list (libradicl + collate.rs) — open
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

### 2.7 LOW — quant loop — open
- `em_optimize_subset` allocates `alphas_in/out` (2 × num_rows) per cell (`em.rs:189-190`);
  `fill_ref_offsets` allocates an nref Vec per cell (`eq_class.rs:285-294`); `EqMap::clear` memsets
  976 k `label_counts` per cell. Keep buffers in the worker.
- Writer mutex holds gzip + featureDump formatting (`quant.rs:2204-2263`); pre-format outside the lock.
- `get_forseti_check_list` rescans `r.refs()` per (read, txp) — fine for small `na`, but positions could be
  carried on the eq-class entry directly once 1.1 is done.

---

## 3. Memory

### 3.1 HIGH — 23.5 GB spliceu transcriptome as `HashMap<u32, Vec<u8>>` (`quant.rs:1521-1538`) — **done 2026-08-24 (forseti-fix20)**
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

### 3.2 HIGH — per-thread, unbounded MLP cache ≈ 7–9 GB — **removed 2026-08-24 with 2.1 (forseti-fix19)**
`forseti.rs:229-234`: `thread_local! HashMap<u64, f64>`; 429 M misses total → 429 M entries across 31
threads × ~20 B (key + f64 + hashbrown overhead) ≈ 8.6 GB, and the same 30-mer is recomputed once per
thread. This is most of the 42 − 35 = 7 GB delta over baseline quant. Fix: process-global shared cache
(`DashMap<u64, f32>`, sharded) or — better — the per-transcript hot-position track of 2.1, which bounds
memory by touched transcripts and removes the hash entirely.

### 3.3 MED — position record = 3 `Vec`s per read (libradicl `record.rs:429-435`) — open
`dirs: Vec<bool>`, `refs: Vec<u32>`, `pos: Vec<u32>` per record (master: 1 Vec). Use one
`SmallVec<[(u32 ref, u32 pos, bool); 8]>` or a single flat `Vec<u32>` with refs/pos interleaved; `dirs`
can be a `u64` bitmask for `na ≤ 64`.

### 3.4 MED — in-memory triplet matrix (`quant.rs:1623-1627, 1656`) [pre-existing] — open
`tmcap = 0.1·num_genes·num_cells` reserved with `num_genes` (not `num_rows`, so USA mode under-reserves
and reallocates three large Vecs under the writer lock). For the 227 k-chunk unfiltered runs this is
multi-GB; `--use-eds` streams instead.

### 3.5 MED — libtorch itself (≈ 0.5–1 GB of shared libraries, OMP pools per worker) → see 2.2. — **gone with 2.2 (2026-08-23)**

### 3.6 OPEN — baseline quant already sits at 33–35 GB on the 227 k-chunk RADs — **resolved 2026-08-21 by 1.4**: the 33 GB was the spliceu FASTA loaded for every resolution; parsimony-em now 6–11 GB on the same inputs (Allen fix15 runs, pbmc_10k)
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

## 5. Suggested order of work (as of 2026-08-19; ~~struck~~ = done)
1. ~~Fix 1.1 (UMI index mismatch) + 1.2 (small cells) + 1.3 (collate buffer)~~ done 2026-08-21/23; 1.1 moved
   ~0.7 % of pbmc_10k entries (real bug), everything since is at float-noise level.
2. ~~`tch::set_num_threads(1)` + `ahash` cache + f32 values + `Vec` instead of `HashMap` for
   `forseti_checking_list`~~ superseded: thread pinning via env (2026-08-11), Vec + tie-break (c0d45fb),
   cache removed by 2.1.
3. ~~Native MLP (drop libtorch) + `include_str!` params.~~ done 2026-08-23 (forseti-fix18).
4. ~~Per-transcript hot-position tracks + hot-only scoring (2.1)~~ **done (fix19)**; replaced 3.2.
5. ~~spliceu storage (3.1)~~ **done (forseti-fix20)**: `.fai` + positional reads, nothing resident.
6. libradicl record / collate allocation work (2.6, 3.3) — benefits all modes. — open
7. ~~Profile baseline quant memory (3.6).~~ explained by 1.4 (2026-08-21).
8. (added 2026-08-24) Profile where fix20's remaining ~4.5 min on pbmc_10k goes (PUG/MCC search vs
   eq-class build) before choosing between 2.4 and 2.7.

## 6. Update log

- 2026-08-19 — review written against a9b2514; reference point Allen TX0029-12 (section 0).
- 2026-08-21 — 1.1, 1.2, 1.4, 2.3 (a19b8ca). pbmc_1k/10k re-quantified; long-read validation unchanged.
- 2026-08-23 — 1.3 collate buffer + `--max-frag-len` check (df228d1); 1.5 deterministic order (c0d45fb);
  2.2 native MLP, libtorch dropped (forseti-fix18). Timing found to be dominated by node type: cbcb30
  (EPYC-9475F) ~1.8× faster than cbcb00-20 (EPYC-7313); all benchmarks since pinned to EPYC-7313.
- 2026-08-24 — 2.1 per-transcript tracks (forseti-fix19): pbmc_10k 16:13 → 6:38, +7 % RSS; U block
  1024 chosen from a 4096/1024/512 sweep; reuse analysis (S < 200 MB, U blocks ≥10× answer ~99 % of queries).
  3.1 `.fai` on-demand sequence (forseti-fix20): RSS 47.7 → 17.5 GB. Validation stack established:
  frozen reference + randomized test, `forseti-shadow` feature, `collated_rad/` gates incl. exact
  pre-EM eq-class comparison. Persisted collated RADs for pbmc_1k/10k.
- 2026-08-25 — pre-submission audit fixes (1c68c8c): worker panic / producer errors abort instead of
  writing partial output, `--spliceu-fa` coverage and `.fai` staleness checks, IUPAC rc, capped
  per-candidate warnings, u32 candidate index, track-store teardown skipped (pbmc_1k 1:06 → 0:59).
  `--forseti-margin` added (063c945, default 0 = unchanged). Resolver `eprintln!`+`exit(1)` sites
  removed, scoring returns plain `Vec` (2b028e2). Gate for both: EQC_IDENTICAL vs fix20 on pbmc_1k and
  pbmc_10k. Hot-path decision / track-reuse counters (Relaxed atomics shared by all workers, ~3 RMW per
  window query) moved behind `--features forseti-stats`; the benchmark build carries none. CHANGELOG
  consolidated. Remaining open items (2.4, 2.6, 2.7, 3.3, 3.4, `do_quantify` merge)
  are deferred past submission; code frozen for the manuscript at this point.
