# Scalability experiments — concise design

## Setup (common to all experiments)

* **Baseline config** (use as default for each OVAT run):
  `layers=2`, `num_neighbors=[25,10]`, `bin_seconds=300` (5 min), `H_global=full`, `sampling=GraphSAGE`, `loss=cross_entropy`.
* **Train protocol:** train from scratch for **3 epochs** per config. Do **not** combine configs.
* **Reproducibility:** fix seeds (`torch`, `numpy`, `random`) and log them.
* **Data splitters:** use the following created snapshots:
```
train_snaps = saved["train_snaps"]
test_snaps = saved["test_snaps"]
scaler_edge = saved["scaler_edge"]
train_ip2idx = saved["train_ip2idx"]
test_ip2idx = saved["test_ip2idx"]
edge_cols_used = saved["edge_cols_used"]
```
* **Record timings & memory** per experiment, loss per epoch.
* **Deliverable:** single dataframe (CSV) with one row per run (one config, one seed) containing the metrics listed below.

## Factors & levels (one factor changed at a time)

1. **Neighborhood fanout**

   * Levels: `baseline [25,10]`, `[10,5]`, `[5,2]`.

2. **Number of layers**

   * Levels: `2 (baseline)`, `1`.

3. **Snapshot bin duration**

   * Levels: `300` (5 min, baseline), `900` (15 min), `3600` (1 hour).

4. **Sparse hidden-state storage**

   * Levels: `full H_global (baseline)`, `sparse cache (store only active nodes; LRU eviction after K bins)`.

5. **Sampling strategy**

   * Levels: `GraphSAGE neighbor sampling (baseline)`, `layer-wise sampling` (sample per layer independently or cap neighbors per node).

6. **Loss function**

   * Levels: `cross_entropy (baseline)`, `asymmetric loss` (specify implementation/hyperparams in run log).

> **Protocol:** For each factor, run one experiment per level (keep all other settings = baseline). **Do not run combined/interaction experiments.**

## Execution order (recommended)

1. Baseline (record)
2. Neighborhood fanout (all levels)
3. Layers (all levels)
4. Bin duration (all levels)
5. Sparse H_global (both levels)
6. Sampling strategy (both levels)
7. Loss function (both levels)

*(Each experiment = 3 epochs.)*

## Metrics to collect (one row per run in dataframe)

**Identifiers / config**

* `config_id`
* `layers`, `num_neighbors`, `bin_seconds`, `H_global_mode`, `sampling`, `loss`

**Performance (edge-level, evaluated on test set after training)**

* `tp`, `fp`, `tn`, `fn` (confusion matrix entries)
* `weighted_f1`, `precision`, `recall`
* `accuracy`, `fpr`, `fnr`
* `train_loss_epoch1/2/3`

**Scalability / resource**

* `peak_gpu_memory_mb` (e.g., `torch.cuda.max_memory_allocated()`), `peak_cpu_memory_mb`
* `time_per_epoch_s` (list or mean), `total_wall_time_s`
* `throughput_edges_per_s` (edges processed / total_train_time)

**Sampling / graph stats**

* `avg_sampled_subgraph_nodes`, `avg_sampled_edges_per_batch`
* `avg_actual_neighbors_available` (per layer)

## Measurement details (one-liners experts can follow)

* **Peak GPU mem:** `torch.cuda.reset_peak_memory_stats(); ...; torch.cuda.max_memory_allocated()`
* **Time:** `t0 = time.time(); train_epoch(...); t1 = time.time(); epoch_time = t1 - t0`
* **Confusion matrix & metrics:** compute on test edges after finishing 3 epochs (use sklearn).
* **Throughput:** `total_edges_processed / total_train_time`.
* **Log everything** (config, seeds, dataset sizes, per-snapshot node/edge counts).

## Output format

* Save a single CSV with the dataframe columns above. Also save per-run JSON logs and model checkpoints.
* This CSV is the only experiment-level deliverable required for later LaTeX reporting.

## Minimal example of `config_id` naming

```
baseline                -> cfg_layers2_neighbors25-10_bin300_full-ht_graphsage_ce
fanout_10-5             -> cfg_layers2_neighbors10-5_bin300_full-ht_graphsage_ce
layers_1                -> cfg_layers1_neighbors25-10_bin300_full-ht_graphsage_ce
bin_900                 -> cfg_layers2_neighbors25-10_bin900_full-ht_graphsage_ce
sparse_h_global         -> cfg_layers2_neighbors25-10_bin300_sparse-ht_graphsage_ce
sampling_layerwise      -> cfg_layers2_neighbors25-10_bin300_full-ht_layerwise_ce
loss_asymmetric         -> cfg_layers2_neighbors25-10_bin300_full-ht_graphsage_asymm
```