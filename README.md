# NuCLR striatal cell types and task responses

This repository connects passive-data pretraining, adaptation to one active animal, putative cell-type assignments, and task-event GLM features. The initial active cohort is animal **745502**, with sessions on December 10, 11, and 18, 2024.

The current biological analysis directions and their connections are described in [ANALYSIS_PLAN.md](docs/ANALYSIS_PLAN.md): cluster–task responses, recorded cell types within functional groups, and anatomical organization. Interesting functional structure is the goal; superiority over simple physiology is not a prerequisite.

## Layout

| Location | Purpose |
| --- | --- |
| `run.py` | One entry point for the workflows below |
| `src/configs/` | Core workflow settings, datasets, cohorts, and model sizes |
| `src/data/` | Shared HDF5 reader, unit/time selection, trial auditing, and sampling views |
| `preprocess/allen_striatum_active/` | [Active Brainsets exporter](preprocess/allen_striatum_active/pipeline.py); raw-source handling stays here |
| `src/models/`, `src/training/` | NuCLR and contrastive training |
| `src/celltypes/` | Cell-type readout, held-out evaluation, and probabilities |
| `src/glm/` | Rich behavioral GLM fitting, cache readers, and local settings |
| `src/embeddings/` | Shared embedding readers, export, and UMAP views |
| `src/experiments/` | Neighbors, clustering, transfer, responses, and neuron tuning; each family keeps its own settings and plots |
| `src/common/` | Checkpoint compatibility, runtime helpers, shared analysis utilities, and plot styling |
| `data/processed/{active,passive}/` | Self-contained Brainsets recording HDF5 files; ignored by version control |
| `results/nuclr/` | Preserved historical runs, checkpoints, embeddings, and plots |
| `results/experiments/` | Outputs from the new workflows |

## Configuration and execution

Use the repository's `venv/bin/python`. Dependencies are listed in `requirements.txt`. Run the commands from the repository root; relative data/output paths also resolve there.

```bash
venv/bin/python run.py --help
venv/bin/python run.py pretrain --show-config
venv/bin/python run.py finetune model=nuclr_48 --show-config
```

`--show-config` only resolves settings. The entry point rejects preparation, fitting, inference, and new UMAP computation on PARC login hosts. Run those operations inside an allocation you have already obtained; this repository never requests an allocation. Redrawing a saved coordinate table is allowed on a login host.

The defaults use all passive sessions and a 128-dimensional NuCLR model. Edit `src/configs/cohort/active_745502.yaml` to change the active sessions, or add another cohort file. `data=active`, `cohort=active_745502`, and `model=nuclr_48` select the corresponding config groups. Other settings use `key=value` overrides. Set worker/thread counts to match your allocation.

Analysis settings live in `src/experiments/<family>/configs/` and `src/glm/configs/`. Commands and short `--config` names resolve these locations automatically. `--config /path/to/settings.yaml` also accepts a saved or custom configuration. The old `glm` and `compare` commands have been removed; use `glm_pilot`, `glm_cache`, or the focused response/tuning workflows for their respective models.

## Export recordings and train

On your existing compute allocation:

```bash
venv/bin/python preprocess/allen_striatum_active/pipeline.py export \
  --raw-root /path/to/active/raw --output data/processed/active --workers 2
venv/bin/python run.py pretrain
```

Preprocessing dependencies are declared in the exporter's Brainsets pipeline header. Export active recordings once with the current schema (2.1.0). Existing processed passive Brainsets recordings go in `data/processed/passive/`. The runtime reads these HDF5 files directly; there is no secondary preparation command, spike binary, or CSV/NumPy recording cache.

The exporter retains available source signals, trial labels, waveforms, electrodes, timing, and provenance. Missing signals stay explicitly unavailable. Source `default_qc` and separately recomputed pre-opto QC remain distinct. No lossy reduction is applied. Runtime analyses do not access the preserved source archive inside the file.

Continuous signals have separate `baseline`, `behavior`, `optotagging`, and `pre_opto` views under `signal_domains`, with domain-specific availability and timestamp quality metadata. `Recording.signal(name, domain="behavior")` returns matching measurements and timestamps on the original source clock; `signal_quality` exposes defects even for unavailable domains. The default `full` view preserves the complete signal and rejects invalid timestamps. Same-file HDF5 virtual datasets avoid duplicating measurements and remain portable when a recording file is moved. A timestamp reversal marks every domain intersecting its time span unavailable; unassignable timestamps or mismatched array lengths remain conservatively unavailable. Behavioral task analyses explicitly read the behavior view.

Selection happens when opening a recording. Active options include `data.domain=pre_opto`, `data.buffer_s`, `data.qc=none|default|pre_opto`, tip-site, sorting, and region filters. Passive defaults use declared prestimulus intervals, exclude stimulation, and preserve gaps. Embedding exports include physiology features calculated over the selected intervals. CSV/NPZ files under results remain derived analysis artifacts, never substitutes for source recordings.

For workstation use, transfer the processed HDF5 directories and any checkpoints/results you need. Set `data.root` for training/export, or `recording_root` for recording-based analyses. Recording IDs use `animal/session/recording`; `NUCLR_ACTIVE_H5_ROOT` and `NUCLR_PASSIVE_H5_ROOT` supply default roots for ID-based access. Raw inputs are needed only to regenerate active exports.

Pretraining prints its run directory and saves `checkpoint.pt` and `embeddings/`. Set the following paths to the actual run outputs before continuing:

```bash
PASSIVE_CHECKPOINT=/absolute/path/to/passive/run/checkpoint.pt
venv/bin/python run.py finetune checkpoint="$PASSIVE_CHECKPOINT"
```

Fine-tuning initializes model weights and starts a fresh optimizer/run on the active cohort. Use `resume=/absolute/path/to/checkpoint.pt` to continue an interrupted run instead. Model dimensions and temporal settings must match the source checkpoint; select `model=nuclr_48` when adapting an existing 48-dimensional model. A 48-dimensional checkpoint cannot initialize a 128-dimensional model directly.

### Sequential training pipeline

`venv/bin/python run.py training_pipeline` runs passive pretraining, active fine-tuning, and final embedding exports inside an existing allocation. Both processed HDF5 directories must already exist. The pipeline does not preprocess raw data or request resources.

Settings are in `src/configs/training_pipeline.yaml`, `pretrain_b200.yaml`, and `finetune_b200.yaml`. The default is 128 dimensions, 50 passive epochs with checkpoints every five epochs, followed by ten epochs on animal 745502 with checkpoints every epoch. Both stages save their final epoch. Data and cohort selections come from the training configs. Checkpoint cadence controls saved model/optimizer states, not repeated embedding exports. Training duration has not been benchmarked.

Each invocation creates a separate directory under `results/experiments/passive_to_active_b200/`, containing `pipeline.json`, resolved stage configs, stage logs, training directories, checkpoints, and embeddings. Fine-tuning receives the completed pretraining checkpoint explicitly. After both stages finish, the pipeline exports active and passive recordings using each final encoder; this supports before/after cell-type readouts in consistent embedding spaces. Cell-type readout fitting is a separate analysis.

Each invocation starts a new pipeline. Saved HDF5-based checkpoints support individual-stage resumption via `resume=...`; a failed export can be rerun independently using its saved config. An abrupt scheduler termination can leave the pipeline status marked running; consult the scheduler state and stage logs in that case.

## Assign putative cell types

After fine-tuning, re-encode the labeled passive neurons with the adapted encoder. Fit the readout in that encoder's space, then apply it to the active embeddings from the same checkpoint:

```bash
ACTIVE_RUN=/absolute/path/to/active/finetuning/run
venv/bin/python run.py export --config export_passive \
  checkpoint="$ACTIVE_RUN/checkpoint.pt" \
  output=results/experiments/active_745502/passive_readout_embeddings

venv/bin/python run.py celltypes \
  embeddings=results/experiments/active_745502/passive_readout_embeddings

venv/bin/python run.py celltypes action=predict \
  embeddings="$ACTIVE_RUN/embeddings" \
  classifier=results/experiments/active_745502/celltypes/classifier.npz \
  cohort=active_745502
```

The default readout uses labeled ChAT cells versus D1/D2 cells grouped as MSN. Unmapped labels are excluded from fitting; untagged cells are never used as confirmed negatives. Predictions are probabilities within these configured classes, not a complete taxonomy. Their probabilities are uncalibrated and class-balanced; `p_ChAT >= 0.8` is an exploratory candidate threshold, not confirmation of identity.

Evaluation leaves out sessions by default (`holdout=animal` is also available), with preprocessing fitted only on the training partition. It compares embeddings against rate, waveform, and ISI features, records unavailable baseline features and skipped folds, and provides comparison metrics on the same scored neurons. This evaluates readout transfer: it does not claim the self-supervised encoder was trained without those sessions. If a training fold lacks a configured class, that fold is skipped and reported.

The same steps can evaluate the passive pretrained encoder before adaptation. Use a separate output directory for each encoder/readout experiment. Embedding-based prediction checks that source and target exports use the same checkpoint. Each fit and prediction saves its settings and provenance. Baseline metadata comes from the embedding export; re-export to obtain features absent from historical exports.

## Fit and compare task responses

The December 10 behavior-only pilot retains full response curves without using embeddings:

```bash
# Run only inside an existing CPU allocation (six CPUs by default).
venv/bin/python run.py glm_pilot
```

Its configuration is `src/glm/configs/glm_pilot.yaml`. It fits striatal SUA and MUA separately at the unit level, excludes noise, and includes cue, factorial choice/outcome, side-specific licking, movement, reward probabilities, recent trial history, and smooth drift. Three folds hold out blocks of trials, with gaps at split boundaries and training-only regularization selection. Choice and outcome share a timing anchor in this session; their contrasts are conditional associations. Drop-group models reuse the full model's selected penalty. Fold spread and positive predictive gains are descriptive reliability measures, not confidence intervals or significance tests.

`venv/bin/python run.py cluster_glm` compares the saved December 10 embedding communities with complete rich-pilot GLM fits, using `src/experiments/clustering/configs/cluster_glm.yaml`. It joins original unit identities, preserves the original cluster colors/names, and saves full curve summaries, held-out feature gains, inclusion audits, and quality-filtered sensitivity plots. Exploratory rank-outcome comparisons adjust for rate, availability, depth, sorting, recorded cell labels and source QC. They describe associations within this recording, not independent prediction or evidence of discrete cell types. Run on an existing CPU allocation; no GLMs or embeddings are refitted.

### Full-recording GLM cache

`venv/bin/python run.py glm_cache` uses `src/glm/configs/glm_cache.yaml` to inventory active Brainsets HDF5 recordings and fit eligible striatal units on an existing CPU allocation. It never allocates a node. Results resume per unit under `results/experiments/glm/active_cache_h5_v2` with source and model identity checks. Final coefficients use every eligible behavioral bin, including bins excluded only by cross-validation split guards. Invalid trials, missing tracking intervals when movement is modeled, and optotagging periods remain excluded.

The root contains `recordings.csv` (coverage and blockers), `unit_index.csv` (filterable unit metadata and paths), `coefficients.csv` (final coefficients only), `reliability.csv` (held-out validation only), and `schema.json` (model identity and interpretation). Individual recordings live under `recordings/<animal>/<session>/<recording>/`, containing `prepared/`, `final/unit_XXXX.npz` plus fit metadata, `validation/units/unit_XXXX.npz` plus validation audits, and `basis.npz`. Final files contain one coefficient vector/intercept and the reconstructed event, probability, history, and available movement curves. Validation files retain three separate coefficient vectors and out-of-fold predictions. Missing-movement models have an explicit `without_movement` variant; their task effects are not movement-adjusted. Missing events are recorded as unavailable, never replaced by zero events. With `allow_missing_anatomy=true`, recordings lacking region annotations retain non-noise units with region explicitly unknown; they must not be assumed to be confirmed striatal units. Annotated recordings remain restricted to ACB/CP. Trials whose recorded reward delivery differs from expected outcome timing by more than 20 ms, and the immediately following history-affected trials, are excluded and recorded. Nonconverged and insufficient-spike fits remain in the index with explicit status.

For downstream analysis, `src.glm.results.select_units` filters the index by animal, session, recording, label, sorting and QC. `load_unit` returns the final model, basis definitions and reliability; validation coefficients are loaded only with `validation=True`. Both default to converged final fits. The coefficients table has one row per global unit ID × feature × basis index, including an intercept row. Coefficients and curves are conditional log-rate effects, not spike counts or coefficient confidence scores. The full-data penalty is selected from pooled inner-validation losses; sparse units without usable tuning use the strongest configured penalty and are explicitly flagged.

`venv/bin/python run.py cluster_glm_cache` compares the existing Sep 12 recording clusters with this cache on an existing CPU allocation. Configuration is in `src/experiments/clustering/configs/cluster_glm_cache.yaml`; outputs are under `results/experiments/cluster_glm/sep12_active_cache_h5_v2`. It writes `recording_availability.csv` into both the GLM cache and comparison output, with independent flags for trials, licking, rewards, movement, anatomy, units, and GLM completion. Use these flags rather than the single primary blocker in `recordings.csv` when conditions overlap. The comparison preserves local cluster identities, uses final kernels, and requires three converged validation folds for held-out feature comparisons. Tables include rate/quality and spike-feature controls, SUA/QC sensitivity checks, and exploratory multiplicity-adjusted associations across recordings. Cluster identities are not aligned between recordings, and these unit-level tests do not establish independent animal-level replication.

`venv/bin/python run.py cluster_task_responses` examines observed spikes and saved held-out predictions around cues, licking-bout onset/offset, and rewarded/unrewarded choices. Its configuration selects March 31 C1/C6 and December 10 C1/C2/C5. It saves equal-bout phase summaries, individual single-unit galleries, fold sign consistency, and matching audits under `results/experiments/cluster_task_responses/sep12_march31_dec10`. Event windows must be entirely eligible and belong to one held-out fold, with identical events across units. Population curves normalize each unit's local rate change by its session mean; raw-Hz curves are retained separately. December 10 also retains model predictions with movement fixed at zero speed. Matches require identical sorting and region plus explicit physiology calipers; unmatched units remain unmatched. These are exploratory comparisons within the recordings used for cluster discovery, not independent cluster validation. Run on an existing CPU allocation; no GLMs or embeddings are refitted.

Results are saved under `results/experiments/glm/745502_2024-12-10_pilot_h5`: `index.html` browses each unit's curves and observed versus held-out predictions; `curves/` retains every fold's curves; `reliability.csv` and `fold_metrics.csv` retain validation measures. `units.csv` preserves source cell-type labels, sorting quality, and anatomy. Fits resume per unit, with configuration and source checks. `stage=plot` redraws completed fits.

## View embeddings

For an observed-spike population overview before interpreting embedding or GLM structure, run `run.py population_psth` inside an existing CPU allocation. The default December 10 analysis uses all raw SUA/MUA units across all recorded regions, groups rows by region, and sorts by stored probe depth. It bins original timestamps at 50 ms around the go cue, then saves raw firing-rate and baseline-relative heatmaps with recorded D1/D2 annotations. Settings are in `src/experiments/responses/configs/population_psth.yaml`; outputs are in `results/experiments/active_function/population_psth_dec10/`. The saved chronological `trial_counts.npy`, `units.csv`, and `trials.csv` support later individual-unit trial heatmaps. This does not fit a GLM or restrict units to those with embeddings.

`run.py region_peak_timing` uses that saved population cache to overlay CP and ACB peak-time densities in the first second after the go cue. It saves an all-unit view, a view of firing increases supported in separate trial folds, and a single-unit sensitivity view. Each unit has equal weight; both regions use the same density bandwidth and unit-area normalization. Peak timing is distinct from response onset. Settings are in `src/experiments/responses/configs/region_peak_timing.yaml`; per-unit results, regional summaries, and figures are saved under `results/experiments/active_function/region_peak_timing_dec10/`. Run inside an existing CPU allocation.

For the simple peak-time comparison with finer bins, `run.py region_peak_timing --config region_peak_timing_20ms` recounts original spike timestamps in 20-ms bins on the same trials, finds one post-cue peak per unit, and overlays CP/ACB densities. It applies no response-strength or significance filter; flat PSTHs have no peak. Outputs are in `results/experiments/active_function/region_peak_timing_dec10_20ms/`, separate from the original 50-ms results.

`run.py region_peak_timing --config region_peak_timing_width` filters broad responses: keep baseline-relative half-height peak widths ≤300 ms, using 20-ms PSTHs with Gaussian smoothing (sigma 20 ms). Peak time is unconstrained within the original 0–1 s search interval; widths are measured on both sides across the complete -2 to +3 s curve. Results in `region_peak_timing_dec10_20ms/short_peak_widths_300ms/` include per-unit widths, selection decisions, selected PSTHs, example curves, and percentages per peak-time bin. This does not measure the duration of lower-amplitude tails or validate response reliability.


`run.py region_peak_timing --config response_width_waveforms` compares those response widths with the source NWB spike-waveform metrics and peak-channel mean waveforms. It saves continuous-width correlations, broad/short response comparisons, waveform plots, and repeated held-out-unit prediction results under `region_peak_timing_dec10_20ms/width_waveform_association/`. Comparisons are repeated by region and for single units; waveform PCA and scaling use training units only. These are within-recording associations with the existing half-height response-width definition, not validation of total response duration or cell type.

`run.py cue_response_glm` fits descriptive go-cue response shapes for all 318 CP/ACB units in December 10, using 20-ms spike counts, a pre-cue baseline, and five overlapping temporal basis functions over 0–1 s. It validates on three contiguous chronological trial thirds, each held out in turn. `results/experiments/active_function/cue_response_glm_dec10/` stores full-data coefficients separately from fold coefficients, held-out prediction metrics, trial assignments, fitted curves, and a PCA view of normalized response shapes. It does not cluster units or replace the rich behavioral GLM cache.

`run.py timing_sorted_psth` reuses the population cache to show CP/ACB responses from 0–1 s in one heatmap ordered by peak time. Rows are scaled by their own largest absolute baseline change; purple/brown region and blue/orange/gray recorded cell-tag strips accompany the matrix. All units are retained, including flat units masked at the bottom. Settings are in `src/experiments/responses/configs/timing_sorted_psth.yaml`; figures, row identities, and arrays are saved under `results/experiments/active_function/timing_sorted_psth_dec10/`. Run inside an existing CPU allocation.

`run.py response_density` shows ACB and CP response distributions in separate panels with identical axes and a shared density color scale, using the original population PSTH's baseline change divided by each unit's mean firing. Both regions use equal unit weights and unit-integral distributions. Settings are in `src/experiments/responses/configs/response_density.yaml`; the focused and full-range figures, inclusion table, and density arrays are saved under `results/experiments/active_function/response_density_dec10/`. Earlier overlay figures remain available. Run inside an existing CPU allocation.

```bash
venv/bin/python run.py plot \
  embeddings="$ACTIVE_RUN/embeddings" \
  predictions=results/experiments/active_745502/celltypes/predictions.csv \
  color=p_ChAT
```

This computes a UMAP from one mean embedding per neuron, using available exported active windows. Add `features=/path/to/features.csv color=column_name` to inspect a GLM feature. The output includes a coordinate table for inexpensive redrawing.

The preserved active-model coordinates can be viewed immediately without loading spikes or a model:

```bash
venv/bin/python run.py plot \
  coordinates=results/nuclr/2026-09-15/155534_335131/nuclr_seed0/plots/cache/umap_session_mean.csv \
  color=label
```

The default plot cohort selects animal 745502. Use `cohort=all` for all recorded neurons in a saved table.

To produce the complete figure set from an embedding export in one pass:

```bash
venv/bin/python run.py plot suite=true cohort=all \
  embeddings=results/nuclr/2026-09-15/155534_335131/nuclr_seed0/embeddings \
  output=results/experiments/umap/active_2026-09-15
```

The suite fits one combined UMAP and colors it by cell type, animal, and session. It also fits a separate UMAP within each animal/session and colors available cell-type labels. Unlabeled units appear in gray. Session-specific layouts are independent; their coordinates cannot be compared directly. The suite retains all exported units, including multiunits, and saves coordinates, mean embeddings, provenance, and a ZIP of the PNG figures.

`venv/bin/python run.py plot --config plot_active_finetuned` applies this to animal 745502's three sessions using the final fine-tuned encoder. It requests `per_session_colorings=[cell_type,animal,session]`, producing three pooled PNGs and three colorings of each independent session fit (12 PNGs), plus an HTML gallery. Animal coloring is uniform for this one-animal cohort, and session coloring is uniform within each individual session. All colorings of a given fit reuse identical coordinates.

## Compare saved embeddings

`run.py glm_embedding_maps` creates side-by-side within-session UMAP/PCA views colored by individual full-recording GLM effects. The default December 10 configuration reuses its existing fine-tuned UMAP and fits PCA to L2-normalized embeddings without GLM supervision. Cue, common choice/outcome, right-minus-left choice, and reward-minus-omission maps share fixed coordinates; additional maps show pre-event, early, and late windows. Outlines distinguish supported parent groups with consistent fold signs, while all matched effects remain visible. Settings are in `src/experiments/responses/configs/glm_embedding_maps.yaml`; PNGs, coordinates, full kernels, PCA parameters, and an HTML report are saved under `results/experiments/active_function/glm_maps_dec10/`. Run inside an existing CPU allocation; no GLM or encoder is refitted.

`run.py active_feature_transfer` extends the individual-feature analysis to 12 GLM terms and four derived choice/outcome contrasts, testing December 10 → 18 and the reverse direction. Each full kernel gets its own two-axis model; scalar probability/history effects use one axis. Models use source-only scaling and five-fold source validation. Frozen transfer and original-embedding neighbor prediction are reported separately, with neuron-bootstrap intervals, SUA evaluation, and joint multiplicity correction across 108 full-feature and fixed-window target tests. Settings are in `src/experiments/transfer/configs/active_feature_transfer.yaml`; outputs are in `results/experiments/active_function/multifeature_dec10_dec18/`. Movement is marked unavailable for transfer because December 18 lacks it. Use an existing CPU allocation; this reuses saved GLMs and embeddings.

`run.py active_outcome_observed` checks the frozen other-session outcome predictions against observed spike counts in December 10 and 18. Rewarded and unrewarded trials are matched by choice side, both programmed probabilities, and nearby trial blocks. Outputs retain reward and omission traces, absolute and relative firing changes, single-unit evaluations, trial-block uncertainty, licking, and an explicitly exploratory common activation/suppression analysis. Settings are in `src/experiments/transfer/configs/active_outcome_observed.yaml`; the report and reusable per-neuron/per-trial data are in `results/experiments/active_function/outcome_observed_dec10_dec18/`. The same recordings supplied the embeddings, so this is observed-spike corroboration rather than independent-data replication. Run only within an existing CPU allocation.

`run.py active_cue_axes` examines one GLM feature: the full go-cue temporal kernel. It learns two linear PLS axes from December 10, checks held-out source neurons, and applies the frozen axes and kernel decoder to December 18. Existing clusters, recorded cell types, anatomy, physiology, and other GLM responses interpret the axes afterward. Source-to-target neighbor prediction is kept separate from within-target neighborhood coherence. Settings are in `src/experiments/transfer/configs/active_cue_axes.yaml`; the report, PNGs, neuron tables, full kernels, validation predictions, and serialized axes are in `results/experiments/active_function/cue_axes_dec10_to_dec18/`. Run only on an existing CPU allocation; GLMs and encoders are reused without refitting.

The new three-encoder comparison is configured by `src/experiments/transfer/configs/encoder_comparison.yaml`: `run.py encoder_comparison` matches original unit identities across historical active-pretrained, passive-pretrained, and fine-tuned exports; compares within-session geometry and label-free clusters; and runs diagnostic unit-held-out identity probes. It saves cluster assignments for downstream analyses. Historical noise-unit population context differs from the new exports, so only passive-pretrained versus fine-tuned isolates adaptation.

`run.py passive_cluster_enrichment` clusters passive recordings independently and tests ChAT versus labeled D1/D2 enrichment, preserving unknown labels. `run.py encoder_cluster_glm` uses the matched active clusters and frozen GLM cache for physiology-adjusted associations and independent cross-session matching. `run.py encoder_shared_prediction` evaluates train-only PCA/regularized CCA and held-out prediction of GLM summaries, with a configurable spline physiology sensitivity. All four workflows require an existing CPU allocation and have matching YAML configs alongside their analysis families under `src/experiments/`. Outputs are organized under `results/experiments/encoder_comparison/`, with full curves, memberships, tables, provenance, and validation records. These are exploratory analyses of encoder-trained recordings, not independent animal-level validation.

`run.py passive_transfer` matches the previously identified passive ChAT-enriched reference group across passive recordings, using source-only centroid/radius rules in each encoder's existing space. Target cell labels evaluate the correspondence and never choose candidates. The behavioral cohort is the intersection of pretrained and fine-tuned decisions; session centering and a firing-rate-only prototype are separate sensitivities. Source-animal exclusion and animal-block bootstrap summaries distinguish across-animal correspondence from within-animal reuse, although these animals were included in encoder pretraining.

Settings are in `src/experiments/transfer/configs/passive_transfer.yaml`; outputs are under `results/experiments/passive_transfer/reference_655572_20230515/`. `candidates.csv` retains all units and selection flags, `target_cluster_matches.csv` relates the matches to existing unsupervised clusters, and `unit_behavior.csv` stores per-unit running/whisking correlations, onset responses, and population coupling. Behavior uses canonical HDF5 signals on the source clock and the same eligible prestimulus intervals as the selected recording. `behavior_availability.csv` explicitly marks missing signals. Rate-matched controls share the recording and sorting category. Comparisons aggregate by animal, with single-unit and population-gap sensitivities; candidates remain provisional functional matches, not verified ChAT labels. `recording_quality.csv` flags internal population-wide spike gaps of at least ten seconds. This quality sensitivity was added after inspecting the first exploratory results; full-data results are retained. The workflow does not infer timing corrections or change existing embeddings.

`stage=match` only computes matching, `stage=behavior` reuses matching and computes behavior plus validation/reporting, `stage=audit` validates saved results and refreshes summaries, and `stage=plot` redraws audited outputs without rereading spikes. All stages run inside an existing CPU allocation. `index.html` and `figures_and_summary_tables.zip` provide the report and exportable summaries; the onset cache preserves full per-unit response curves.

`run.py active_choice_transfer` begins the active cluster–task direction with a frozen December 10 → December 18 readout in animal 745502. It retains full GLM right-minus-left contrast curves, including separate rewarded and unrewarded contrasts, and displays them by existing unsupervised cluster. Source-only nested unit validation selects a ridge penalty; target GLM features never choose transforms, penalties, or cluster matches. A source-cluster profile readout and predefined session-centering sensitivity accompany the primary fine-tuned/original-space readout. Recorded cell types and anatomy are retained for interpretation, not fitting. The feature is a conditional choice-associated response, not trial-choice decoding or reward prediction error. Settings are in `src/experiments/transfer/configs/active_choice_transfer.yaml`; `results/experiments/active_function/choice_dec10_to_dec18/` contains the report, per-neuron features and predictions, full observed/predicted curves, serialized readouts, and validation records. Run on an existing CPU allocation; it does not refit GLMs or encoders.

`venv/bin/python run.py joint_umap` uses `src/configs/joint_umap.yaml` to visualize the new fine-tuned encoder on passive and active exports together. It verifies matching encoder signatures, averages available windows per recorded unit, and fits one cosine UMAP shared by the animal- and session-colored PNGs. Active units use outlined diamonds; passive units use circles. Two independent passive-session maps are selected by ChAT-label count, preferring different animals. All exported single units and multiunits remain included. Results include reusable mean embeddings, coordinates, session counts, source provenance, an HTML gallery, and a PNG ZIP. Run on an existing CPU allocation; this does not train a model or infer cell types. Reusing the same output resumes cached means and combined coordinates only when configuration and export provenance match.

`venv/bin/python run.py cluster` uses the cached Sep 12 mean embeddings to cluster each recording independently on an existing CPU allocation. Settings are in `src/experiments/clustering/configs/cluster.yaml`. It builds cosine nearest-neighbor graphs in the original 48 dimensions and finds Louvain communities, excluding noise labels. Checks vary neighborhood size and random seed and repeat clustering with SUA only. The workflow compares waveform scalars and firing-rate/ISI summaries recomputed over the exported recording span, preserves missing features, and saves per-recording figures, unit assignments, feature effect sizes and exploratory corrected tests. Waveform associations are also checked after accounting for rate, sorting, depth and embedding availability. Saved UMAP coordinates are used only for display; cluster colors are recording-specific. Graph partitions and physiological differences do not establish cholinergic identity. Interrupted runs can resume cached feature extraction with identical settings; completed outputs are not overwritten.

To inspect labeled-neighbor agreement over time for December 10 using the September 12 export:

```bash
venv/bin/python run.py neighbors
```

Each labeled unit is a fixed heatmap row. At each 10-second window, the workflow finds its ten nearest other D1/D2-labeled units by cosine distance in the original 48D embeddings. Colors show the fraction sharing the query's class: dark red at zero, dark green at one. Missing queries or fewer than ten valid neighbors appear gray. All labeled sorting categories are included. The output retains exact neighbor identities and scores for inspection.

The neighbor-pair, pair-trace, task-association, pair-diagnostic, and task-neighbor plotting implementations remain under `src/experiments/neighbors/`. Their old session-specific presets referenced missing inputs and have been removed. To reuse these analyses, supply a complete configuration explicitly, for example `run.py neighbor_pairs --config /path/to/settings.yaml`. The task-neighbor plot is also available as `python -m src.experiments.neighbors.plots.task_neighbors --config /path/to/settings.yaml`.

`venv/bin/python run.py distance_task` screens direct cosine distances between all unique pairs of units present in >75% of the December 10 windows, including unlabeled units. `src/experiments/neighbors/configs/distance_task.yaml` configures the cohort and output. Selection uses training windows only; full one-minute blocks every four minutes are held out, with 30-second guards. Models control slow time trends, nonlinear terms of both firing rates, movement, availability and mean population distance excluding the pair. Candidates are evaluated by held-out error reduction for each task group, with predefined population-control and spike-pattern sensitivities. Window-level ISI/Fano features require at least ten spikes. The existing encoder was pretrained on this session: validation concerns within-session task associations, not encoder generalization. This CPU workflow uses existing exports and does not train or infer a new model. Stages `discover` and `validate` reuse the prepared output when explicitly requested.

For a basic linear comparison of session and animal identity in the two saved embedding sets:

```bash
venv/bin/python run.py probe
```

This uses the mean embeddings cached by the UMAP suites, keeping the original 48 dimensions. It matches the same units across exports and uses identical neuron splits, training-only standardization, and a fixed regularized logistic readout. It also tests animal decoding with entire sessions held out from readout fitting. These are readout evaluations on cohorts already used for encoder pretraining. Outputs are saved under `results/experiments/identity_probe/sep12_vs_sep15`.

## Preserved results and removed caches

Historical `results/nuclr/` contents are preserved as recorded. New recording-based work uses HDF5 files. PSTH and tuning workflows read trials directly through `src/data/task.py`; running a GLM first is unnecessary. Analyses of fitted GLM properties still require their corresponding model results.

Historical configs still describe their original locations. For a new export from a preserved active checkpoint, use `--config export_active` and select the desired HDF5 cohort explicitly. Old model weights can initialize new runs when architecture and sampling match; exact resumption of legacy cache-based runs is not supported. New checkpoints resume against their saved HDF5 selection fingerprint; `data_root=/new/location` changes the storage location without changing selection semantics. Existing active HDF5 exports from older schemas must be regenerated.

New embedding manifests retain the complete window schedule, including silent windows and gaps between eligible intervals. Analyses that align embeddings with recording time require these new exports. Historical embeddings remain usable for analyses that need only the saved vectors and metadata.

Active preprocessing and validation ran on CPU job `9409497`: all 48 recordings across 23 sessions and four animals were regenerated with schema 2.1.0 in `/vast/projects/dyer1/lab/shared/neural_data/allen_striatum/processed`. The local `data/processed/active` link points there. All files passed source-array checksum, canonical spike/event/trial, domain-view, and repository unit/trial loading checks. A December 10 training window and behavior-speed binning also passed. April 9's 63 affected tracking signals are usable during behavior and pre-opto; their four backward timestamp jumps remain documented in the unavailable optotagging and full-recording views. Domain-boundary, empty/missing-domain, and file-relocation checks passed. Full training and analysis workflows have not been run as part of this validation.
