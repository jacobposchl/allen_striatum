# Neuron tuning profiles

## Goal and scope

For each neuron, collect a profile of how its activity relates to four axes:

1. Programmed reward probability at each spout.
2. Running and licking.
3. Choice and outcome around individual trial events.
4. Recent reward experience at each spout.

These features describe associations. They can overlap, and they do not need to
come from one model. A neuron can have several associations, and those
associations can change during a session. The profile should retain those changes
instead of forcing every neuron into a fixed tuned/not-tuned category.

This document specifies a new analysis. The existing GLM cache is a separate
model; its coefficients are not results from the methodology below. The initial
recording is animal 745502, session 2024-12-10_12-06-14,
experiment1_recording1. Use the behavioral task period, excluding optotagging.
Axes B and C are implemented for this session. The other axes remain
methodology specifications; the existing GLM coefficients do not substitute for them.

## Shared data and feature records

Use original spike times, trial identities, registered choices and outcomes,
programmed probabilities, detected licks, and running speed. Preserve the
original unit ID, animal, session, region, sorting category, and source quality
flags. Treat single units and multiunits separately in population summaries.
Embeddings and cell-type labels are not predictors in these analyses.

Carry forward the documented timing and reward-delivery audits. Exclude invalid
task intervals and uncertain outcomes where they affect a measurement. Keep
original chronological trial positions when calculating history. Missing
behavior is missing, not zero behavior.

Use axis-specific eligibility masks. For example, missing speed does not make a
probability measurement unusable when the spikes and trial probabilities remain
valid. Reuse audited inputs where appropriate, but do not automatically inherit
the old GLM's movement-dependent mask or validation boundary exclusions.

Each saved feature should include its estimate, units, relevant spout or
condition, time window, supporting observations and duration, uncertainty or
sensitivity summary, and any missing-data reason. Save underlying response
curves and observations alongside scalar features. Undefined estimates remain
missing rather than becoming zero.

## Validation approach

Validation is axis-specific. **Axis C now uses three-fold held-out prediction
and curve stability as its primary quantitative checks**, as detailed below;
modeling changes in tuning is deferred, while chronological prediction errors
are retained for later investigation. For the other axes, the primary checks
remain support, uncertainty, influence, and variation over time. Held-out
prediction does not determine whether their descriptive features are retained.

- **Support:** How many distinct blocks, trials, bouts, spikes, and seconds
  contribute? Are the conditions needed for the comparison actually present?
- **Uncertainty:** How much does the estimate vary under resampling that respects
  temporal dependence? Adjacent time bins are not independent replicates.
- **Influence:** Does removing one block or one contiguous stretch change the
  estimate substantially? This is a sensitivity check, not a held-out prediction
  test.
- **Variation over time:** Estimate the same feature in predefined chronological
  portions of the session where support permits. Differences may reflect real
  changes, sampling uncertainty, or changes in behavioral coverage; distinguish
  these possibilities rather than requiring agreement.
- **Agreement with observations:** Show the actual block measurements, behavioral
  response curves, and event-aligned spike responses used to estimate features.
  For fitted models this is a descriptive fit check, not independent validation.

For axes using a first time-resolved view, use early, middle, and late thirds of elapsed
task time, chosen without inspecting neural responses. Keep unsupported
comparisons missing. Do not search for the interval with the strongest apparent
tuning and then treat that interval as independently confirmed.

Resample contiguous units of observation, not individual bins. Within a
reasonably stable epoch, moving blocks can provide approximate uncertainty
bands. Choose resampling lengths to preserve the relevant dependence and inspect
sensitivity to that choice. A whole-session bootstrap does not automatically
handle changing tuning: when changes are substantial, emphasize epoch-specific
estimates and observed variability instead of a single confidence interval.
Bootstrap stationarity assumptions are discussed by
[Politis and Romano (1994)](https://doi.org/10.1080/01621459.1994.10476870).

Uncertainty bands, deletion sensitivity, temporal variation, and null-hypothesis
tests answer different questions. Label them separately. In-sample model fit or
a narrow uncertainty band alone does not establish unique encoding or causation.

## A. Programmed reward probability

### Question

Does overall firing vary with the programmed probability at the left spout or
at the right spout? Can a neuron prefer an intermediate probability level?

### Observations and calculation

Construct each spout's blocks independently. A left block is a maximal
consecutive run with unchanged left probability; changes on the right do not
split it. Apply the same rule separately on the right. Exclusions remove exposure
from a block without manufacturing a new probability block.

For neuron n and a block b, calculate:

`block firing rate = spikes in usable block time / usable block duration`.

Use the whole eligible task period within the block, including intertrial time
where task context is defined. No pre-cue restriction is required. Retain the
block's probability, duration, spike count, trial count, chronological location,
and the other spout's probabilities during that block.

For the initial session, the left and right probabilities each change 16 times,
giving 17 blocks per spout before eligibility checks. The available probability
levels are 0.1, 0.4, and 0.7. These are independent block partitions, not 34
independent experimental replicates.

### Features

Calculate the following separately for left and right:

| Feature | Definition |
| --- | --- |
| Probability-response profile | Mean block firing at each probability level, retaining every contributing block rate. |
| Probability association | Pearson correlation between block firing and probability; undefined when either has no variation. |
| Level-specific preference | Mean firing at one probability minus the equally weighted mean at the other two probabilities. |
| High-minus-low contrast | Mean firing at 0.7 minus mean firing at 0.1, in Hz. |

Let `m_s(p)` be the mean of block firing rates for spout s at probability p. For
example:

`preference_left(0.4) = m_left(0.4) - [m_left(0.1) + m_left(0.7)] / 2`.

A positive value captures elevated firing specifically at 0.4; a negative value
captures suppression. Save preferences at all three levels. These contrasts are
related and intentionally redundant. Do not invent a three-level contrast when
a required level is absent.

The primary profile gives each eligible block equal weight, so a long block does
not become many replicates. Also show duration-weighted estimates as a
sensitivity check and retain block duration to identify very short blocks.

### Validation

1. Show individual block rates, labeled by probability and ordered in time.
   Report the number of blocks at every probability. One block at a level cannot
   establish repetition across blocks, however many spikes it contains.
2. Delete one block at a time and recompute profiles and contrasts. Report the
   resulting range and whether sign or preferred level changes. If deletion
   removes a required level, report that comparison as unavailable.
3. Compare equal-block and duration-weighted results and inspect the influence
   of short or low-exposure blocks.
4. Compare repeated occurrences of the same probability and locally available
   level contrasts across session epochs. Report the supported contrasts only.
5. Where enough repeated blocks exist within a reasonably stable period,
   resample contiguous runs of paired block rates and probabilities to estimate
   uncertainty. Report missing-level bootstrap samples and limited support;
   never treat bins within a block as independent probability repetitions.

A probability that occurs only in one portion of the session cannot be cleanly
separated from a time-related change in firing using these observations alone.
Keep the association, flag that limitation, and do not silently regress it away.
Likewise, left and right associations remain marginal descriptions and may
reflect correlated behavior or the other spout's probability.

## B. Movement and licking

### Question

How does firing change during running bouts, track speed within those bouts,
and covary with licking at each spout?

### Observations and calculation

Use 100-ms bins for spike counts and mean absolute wheel velocity. Include all
detected left/right licks, including registered choice licks. For licking,
estimate firing and lick rates with the same centered 500-ms boxcar, evaluated
every 100 ms. The two spouts remain separate.

Use only bins with valid measurements for the behavior being examined. Keep
true zero-speed and zero-lick bins. Do not interpolate across long missing
intervals or smooth across recording gaps. Ordinary trial boundaries do not
create missing time. Ambiguous trial intervals retain their audited exclusion
padding. A speed bin requires at least 80% of expected native samples.

Detect running using behavior alone: smooth speed with a Gaussian of sigma
0.2 s, enter above 1 cm/s, and exit below 0.5 cm/s. Merge interruptions of at
most 1 s if the intervening data are observed; retain bouts lasting at least
2 s, with no maximum duration. The quiet-speed median and robust standard
deviation are both zero in this session, so an SD-only threshold is unsuitable.
These physical thresholds are starting settings, with sensitivity checks below.

For each bout, use quiet samples at or below 0.5 cm/s between 5 and 1 s before
onset as its baseline, requiring at least 2 s of quiet exposure. A bout without
a baseline can still contribute a within-bout speed correlation. Give supported
bouts equal weight in the neuron-level summary.

### Features

| Feature | Definition |
| --- | --- |
| Running response | Mean firing during each bout minus its quiet baseline, in Hz; save mean, median, and fraction of positive bout responses. |
| Running modulation | Per-bout `(running − quiet) / (running + quiet)`, then average across supported bouts; undefined if both rates are zero. |
| Within-bout speed association | Pearson r between unsmoothed 100-ms firing and speed, then mean r across supported bouts; require speed SD of at least 0.25 cm/s and nonconstant firing. |
| Running onset profile | Mean baseline-subtracted firing from −5 to +10 s, with sigma 0.2-s display smoothing; save individual bouts and sample support. Mask neighboring qualifying bouts. |
| Onset magnitude and timing | Signed maximum absolute response in 0–10 s and its time, descriptive only; no precise latency or significance claim. |
| Response persistence | First- and last-third firing changes relative to the same quiet baseline, their difference, and a 20-bin duration-normalized response curve. |
| Duration association | Correlation of bout duration with mean baseline-subtracted firing, requiring at least five supported bouts; use mean firing, not total spikes. |
| Left/right licking association | Separate Pearson correlations between 500-ms firing and lick rates; zero lag is primary. |
| Left/right licking profiles | Mean observed firing in each lick-rate category, with exposure saved; no licking, 2, 4, 6, 8, 10, and >10 Hz categories. |
| Secondary lag profiles | Running and each spout at −0.5, 0, and +0.5 s, with identical eligible anchor samples across lags. |
| Supplementary continuous running features | Whole-session speed correlation and firing profile by speed range, retaining quiet time. |

The paired quiet/running means use the same baseline-supported bouts. The
separately saved mean running rate can use every bout. Within-bout correlations
use only supported bouts for each neuron; save their number rather than treating
undefined values as zero. No peak lag is selected from the three lag checks.

Define lag using `corr(behavior(t), firing(t + lag))`: positive lag means firing
follows behavior. Running lag comparisons keep all pairs within their own bout;
their common-support zero-lag value can differ from the primary value that uses
the entire bout. Licking validity includes the complete smoothing and lag
support. The 500-ms smoothing limits how precisely lick timing can be interpreted.

### Validation

1. **Support:** Save bout durations, baseline availability, eligible seconds,
   behavioral occupancy, spike counts, and the number of contributing bouts
   for each feature. Flag fewer than 50 spikes; suppress running resampling
   bands with fewer than five supported bouts. Show lick-profile categories
   only when they contain at least 5 s of exposure.
2. **Descriptive resampling:** Use 400 paired resamples. Running resamples groups
   of three consecutive bouts for mean running response, within-bout r, and
   persistence. Licking resamples 60-s blocks with at least 80% paired coverage,
   recomputing the pooled correlation from those blocks. Save the corresponding
   block-supported point estimate separately from the full-session r.
3. **Influence:** Delete one bout or one licking time block at a time and save
   the resulting range. This asks whether an observation dominates the estimate.
4. **Variation over time:** Save individual bout responses, local 60-s licking
   correlations, and early/middle/late feature summaries. Missing local
   correlations remain missing when there are no licks or no firing variation.
5. **Running threshold sensitivity:** Repeat detection and baseline selection
   with thresholds multiplied by 0.75 and 1.25, retaining the minimum duration.
   Save both the change in bout support and the resulting mean response.

The resampling bands describe sensitivity under the specified grouping and
local-stability assumptions. They are not significance tests or guaranteed
confidence intervals under nonstationarity. The current implementation does
not assign formal uncertainty to peak onset timing or categorical response
curves. Smoothing-scale and resampling-block-length sensitivity remain possible
follow-ups, not completed validation. No held-out acceptance gate is used.

Task responses can contribute to behavior associations. We retain that overlap
in the primary fingerprint; an optional comparison within task conditions asks
a narrower question and should be labeled separately.

### Computed session outputs

`run.py movement_licking` uses `src/experiments/tuning/configs/movement_licking.yaml` and requires an
existing compute allocation; it never requests a new node. The first result is
`results/experiments/neuron_tuning_profile/745502_2024-12-10/movement_licking_v1/`.
It contains 318 units (106 SUA, 212 MUA), 36 detected running bouts, and 30 bouts
with quiet baselines. Eligible task time contains 1,750 left and 1,675 right licks.

`features.csv` contains scalar features and support, `unit_bout_features.csv`
retains each bout response, `epoch_features.csv` and `lick_block_correlations.csv`
retain temporal variation, and `running_threshold_sensitivity.csv` records
threshold effects. `profiles.npz` retains response curves and their support;
`behavior.npz`, `config.yaml`, and `method.json` record inputs and definitions.
The HTML overview links running, licking, and temporal figures in PNG/PDF/SVG.
The running examples retain units 382, 237, and 132 from the earlier view;
licking examples are the three QC-passing SUA with the largest absolute
left/right zero-lag association. Both selections are descriptive.

### Review of support and sensitivity

`run.py movement_licking_review` audits the saved result without changing its
continuous feature estimates. The first review is saved in the sibling
`movement_licking_review_v1/` directory. It uses 1,000 resamples and checks:

- Running resampling groups of 1, 3, and 5 consecutive bouts; 3 and 5 enter the
  candidate screen. Detection-threshold and deletion checks reuse the original
  saved results. They must preserve the direction of a candidate response.
- Within-bout correlations at 100 and 500 ms, using the same truncated bout
  exposure, and after requiring at least 5 or 10 spikes within a bout. The
  candidate screen requires at least 10 bouts with at least 5 spikes and
  agreement in direction after rebinning and the spike-support restriction.
- Licking estimates with 300-, 500-, and 700-ms windows on common valid support;
  resampling blocks of 30, 60, and 120 s; and deletion of individual blocks.
  Candidates require agreement in direction and at least 10 minute blocks with
  defined correlations. The size cutoff applies to the original 500-ms estimate,
  so borderline candidates can cross that cutoff with other window widths.
- Licking correlations after subtracting each 60-s block's mean from both
  signals. This diagnoses slower cofluctuation; it does not isolate licking
  from reward, choice, or other movement.
- Temporal summaries, overall firing rates by epoch, and resampled peak timing
  and bout-duration correlations. Temporal change is retained, not a rejection
  criterion. Whole-session firing changes alone cannot distinguish changing
  tuning from changes in recording stability; waveform stability was not audited.

There are 72 single units passing the saved default quality flag and minimum
spike support (42 CP, 30 ACB). Another 65 multiunits pass those checks, but are
kept separate from individual-neuron counts. Running occupies 216 s in 36 bouts;
30 have usable quiet baselines, split 9/2/19 across early/middle/late thirds.
Licking has 72 eligible minute blocks, including 68 with left licking and 64
with right licking. These blocks and bouts are not claimed to be independent.

For an explicit descriptive screen, use absolute mean running change at least
1 Hz or absolute correlation at least 0.2, plus the checks above. This identifies
27 running-response, 10 within-bout-speed, 3 left-licking, and 8 right-licking
candidate single neurons, with 36 unique neurons across those overlapping sets.
The lower thresholds of 0.5 Hz or |r| = 0.1 give 36/24/12/20 candidates; the higher
thresholds of 2 Hz or |r| = 0.3 give 19/6/2/3. These are review cutoffs, not
biological boundaries, calibrated tests, or a replacement for continuous features.

Of the 27 running-response candidates, 26 have the same response direction early
and late. All licking candidates retain their direction in all three epochs.
Amplitudes can still change substantially, and the middle-third running estimate
has only two paired observations. Among running-response candidates, the median
span of the 95% descriptive peak-time resampling range is 3.4 s; precise peak
timing should therefore remain secondary even when the firing-rate response is
well supported. Narrow resampling ranges do not include all timing errors from
the detector, smoothing, or acquisition clock.

Use the core features for an exploratory fingerprint of this session, carrying
support and temporal summaries with them. Keep low-support timing, duration,
and high-lick-rate details exploratory. This review does not establish a stable
cell trait across recordings or animals and does not calibrate false-discovery
rates across the many features. `unit_review.csv` retains all units;
`candidate_neurons.csv` is a compact inspection table, not the recommended
input-only subset for clustering or other population analyses.

### First exploratory figure: rolling running-speed correlations

`run.py running_correlation` generates the December 10 heatmaps from saved,
audited spike counts and speed. Run it only inside an existing compute
allocation. Settings are in `src/experiments/tuning/configs/running_correlation.yaml`; outputs are
under `results/experiments/neuron_tuning_profile/745502_2024-12-10/running_correlation/`.

The initial view uses 100-ms bins, 120-s centered windows stepped by 10 s, and
lag panels at -0.5, 0, and +0.5 s. Positive lag means firing follows running.
All panels use identical valid time pairs, unit order, and a -1 to +1 color
scale. At least 80% of a window's pairs must be eligible; constant speed or
firing produces an undefined correlation. Missing windows are gray. Rows
include CP/ACB SUA and MUA, ordered by region and depth, with their original
identities saved in `units.csv`. This view does not filter on GLM fit success.

The saved arrays, window coverage, configuration, and method record accompany
PNG/PDF/SVG figures. These overlapping-window correlations are descriptive, not
significance tests. The initial settings are visualization choices, not fixed
requirements for the remaining movement-axis features.

`run.py running_examples` adds a final-30-minute view: the existing zero-lag
heatmap beside signed source velocity, derived acceleration, and three observed
firing traces. Examples are the three default-QC-passing single units with the
largest median local correlation, requiring at least 80% window coverage and
median r >= 0.2. Selection uses only complete 120-s windows inside the displayed
interval and is descriptive, not independent validation. The heatmap continues
to use absolute speed; the velocity trace retains direction. Display traces use
a Gaussian with sigma 1 s within eligible segments. Acceleration is the
derivative of smoothed velocity, with segment-edge estimates excluded. Selected
identities, the full selection audit, arrays, and methods accompany the figure.

`run.py running_bouts` focuses that view on five isolated bouts selected from
behavior alone, keeping the same three example neurons. The initial detector
uses absolute velocity above 1 cm/s after 0.2-s Gaussian smoothing, merges gaps
up to 1 s, and retains 2–15 s bouts with an isolated -5 to +20 s display window.
It takes the first five qualifying bouts chronologically. The left panel is an
actual run-onset population PSTH averaged across those bouts, relative to a
-5 to -1 s baseline and scaled by each row's largest absolute change. Raw Hz
curves remain saved. The right panels show signed velocity and observed firing
in Hz with common scales across bouts, using lines only. Continuous task
recordings are retained across ordinary trial boundaries; actual excluded
intervals and missing velocity remain excluded. The figure uses the same 1-s
Gaussian display smoothing and is descriptive rather than independent evidence
of tuning. Settings and the complete bout-selection audit are saved with it.

## C. Trial-event responses

### Question

How does the response around registered choice depend on choice side, reward
versus omission, and their interaction?

### Model and calculation

Use one registered-choice alignment. Fit each neuron separately with a Poisson
log-link model for binned spike counts. The initial model contains an intercept
and four event terms:

- Common task-event response.
- Choice side: +0.5 for left and -0.5 for right.
- Outcome: +0.5 for reward and -0.5 for omission.
- Side × outcome: the product of those two contrasts.

Use **five temporal basis functions for every event term**, with identical shapes
and support across terms and neurons. There is no slow-drift predictor in the
initial model, and no separate cue, movement, probability, or history predictor.
All eligible choice events contribute on the continuous time axis; overlapping
event contributions add rather than duplicating spike observations in separate
trial windows.

Writing the four kernels as K0, KS, KO, and KSO, the condition kernel is:

`K(side, outcome, lag) = K0(lag) + side * KS(lag) + outcome * KO(lag)
                       + side * outcome * KSO(lag)`.

The intercept remains an unpenalized baseline. If ridge regularization is used,
fix the scaling and selection rule before producing features, and apply them
consistently across terms. Do not select a penalty because it produces a desired
response or declare tuning from nonzero ridge coefficients. Penalty choice is a
model setting, not a validation outcome.

Five functions over a broad window yield smooth curves; the spike-bin width is
not the effective resolution of estimated response timing.

### Features

| Feature | Definition |
| --- | --- |
| Four condition curves | Left rewarded, left omitted, right rewarded, and right omitted. |
| Common response | Equal-weight mean of the four condition kernels, K0. |
| Side contrast | Left minus right, equally averaged over outcomes, KS. |
| Outcome contrast | Reward minus omission, equally averaged over sides, KO. |
| Interaction | Left reward-minus-omission contrast minus the corresponding right contrast, KSO. |
| Mean signed effect | Mean of each kernel or contrast over a predefined summary window. |
| Peak effect and time | Signed largest-absolute effect and its time relative to registered choice. |
| Main-response duration | Width of the contiguous same-sign lobe around the peak above half of that peak's absolute magnitude. |

Save curves and contrasts in log-rate units. Exponentiated contrasts are rate
ratios. Averaging log-rate kernels is not averaging firing rates in Hz; observed
condition firing rates in Hz are retained separately. Flag peaks at window
boundaries, truncated durations, multiple lobes, and unstable timing. Avoid
interpreting timing for a nearly flat response.

### Validation

Use the agreed **three-fold prediction protocol**, with one set of session-wide
kernels. Do not model changes in tuning over session time in this version.

1. Partition all stored trials chronologically into **nine** near-equal blocks;
   divide each into three consecutive near-equal sub-blocks A/B/C. Define these
   groups before applying exclusions and without inspecting spikes. Fold A
   trains on B+C across all nine blocks and tests A; rotate for the other folds.
   For this session, the test folds have 176, 175, and 175 eligible choices.
2. Preserve complete -1 to +2 s choice windows within one fold. Move nominal
   time boundaries into the gaps between neighboring choice windows when needed.
   Exclude 1 s of **training** time next to each train/test boundary. Preserve all
   valid test time and all eligible choice windows; final all-data fitting has
   no validation guards. Keep original time and actual exposure throughout.
   These guards reduce short-range proximity; they do not remove slow dependence.
3. Fit three models independently using each fold's training counts: the full
   factorial model; a shared-event model with just intercept plus five common
   bases; and a constant-rate model with one intercept fitted to all training
   time. Full and common models use the same fixed coefficient prior. Neither
   fitting nor initialization uses held-out spikes or the final all-data fit.
4. Evaluate **Poisson log-likelihood improvement** on excluded trials. Primary
   scores cover the choice windows; save additional scores for each condition
   and all valid task time. Compare full versus constant, full versus common,
   and common versus constant. Report raw likelihood differences, bits/spike,
   bits/second, and observed/predicted spike totals. Pool by summing likelihood
   differences and dividing by total held-out spikes, not by averaging fold
   ratios. Zero-spike test groups have undefined bits/spike but retain raw gains
   and bits/second where models are estimable. Failed fits remain missing.
5. Quantify robustness on the common grid of four predicted condition curves:
   mean pairwise RMS difference in Hz across the three fits, normalized by the
   RMS of their mean curve. Save per-condition values and secondary shape
   correlations (undefined for flat curves). This describes sensitivity to
   training-trial selection; overlapping training sets do not provide three
   independent estimates or a formal confidence interval.
6. Retain the prediction scores for all **27 original sub-blocks**, in time
   order. Poor local prediction can motivate a later investigation of tuning
   changes, omitted predictors, or recording quality; it does not identify the
   cause. Do not automatically reject a neuron because one fold scores poorly.
7. Refit the fixed full model using every eligible bin for final descriptive
   curves. Use the quantitative population overview and tables first; individual
   condition-curve overlays are a final targeted visualization check.

Positive gain is descriptive predictive improvement relative to the stated
reference, not a significance test, causal effect, or binary tuning threshold.
This protocol evaluates omitted stretches within the same recording; it does
not establish prediction in a future session. No bootstrap, block-deletion,
local tuning fits, or regularization sweep is included in this run. Future
regularization selection must use only the outer-fold training data.

The `v2` output includes `fold_audit.csv`, `subblock_audit.csv`, `fold_scores.csv`,
`pooled_scores.csv`, `subblock_scores.csv`, `curve_stability.csv`, `unit_summary.csv`,
and `fit_audit.csv`. `profiles.npz` stores final and fold-specific coefficients,
condition curves, observed rates, held-out predicted rates, and support.
`validation_overview.png` and `index.html` provide the population review.

### Current trial-filter revision

The trial-response GLM uses **current-trial eligibility**. A trial is not removed
merely because the previous trial had an ambiguous reward record: this model
does not use reward-history predictors. Restore a history-only exclusion only
when that trial's own task labels and timing pass the source audit requirements.
Direct reward ambiguities, invalid records, and their existing time padding
remain excluded. Apply this revised eligibility both to choice-event selection
and to the excluded-time mask. Preserve the original eligibility and exclusion
reason beside the revised decision in `trial_audit.csv`.

For 745502 on December 10, restore trials **84, 110, and 213**; retain exclusions
for **83, 109, and 212**, which contain unmatched water-delivery events. This gives
533 usable stored trials and **526 eligible choices**: 129 left rewarded,
190 left omitted, 120 right rewarded, and 87 right omitted. The seven no-response
trials contribute baseline time. The remaining exclusions keep their padding
of 1 s before and 2 s after the invalid trial. Consequently, trial 110 loses
about 0.666 s of its pre-choice window to trial 109's padding, while its choice
and post-choice response remain available; lag-specific exposure is retained.

The revised configuration writes to `trial_event_glm_v2/`. Existing `v1` fits and
figures retain the old filtering and must not be reused as corrected results.
The trial-table audit is checked separately; fitting and plotting still require
an existing compute allocation. No new allocation is requested automatically.

Keep the current prior SD of 2 fixed for this first run of the agreed validation
protocol. Further comparisons of regularization strengths, including no
regularization, remain deferred. The old bootstrap, block-deletion, and epoch-fit implementation has been removed.
Prioritize the four condition curves when reviewing the fit; 0–1 s scalar
summaries are secondary.

The earlier `trial_event_glm_v1/` outputs are historical records with different trial filtering and validation. They are not corrected v2 results. The current implementation supports only `threefold_subblocks_v1`, configured in `src/experiments/tuning/configs/trial_event_glm.yaml`.

## D. Recent reward experience

### Question

How does firing vary with the fraction of recent choices that were rewarded,
separately at the left and right spouts and at different history lengths?

### Observations and calculation

Use **5, 10, and 20 preceding task trials**. These are chronological trial
windows, not the last 5, 10, or 20 choices at a particular spout.

For current trial i, spout s, and window W:

`H(s, W, i) = rewarded choices at s in trials [i-W, ..., i-1]
              / all choices at s in trials [i-W, ..., i-1]`.

The current outcome is excluded. An unchosen spout contributes neither a reward
nor an omission. If there are no choices at that spout, H is missing. Require a
full W-trial history; initial trials without it are missing for that window.
If a choice/outcome record in the window is ambiguous, mark the affected history
unavailable rather than silently treating uncertainty as an omission or
compressing the chronological window.

Attach H to eligible time bins of the current trial and construct a response
profile: total spikes divided by total exposure within each recent-reward-
fraction range. This is a profile over reward experience, not a separate
correlation of current-trial mean firing with history. It uses the full eligible
trial period and does not impose a pre-cue restriction.

Use common fraction ranges across neurons and windows. Retain exact fractions
and their denominators because a fraction of 1 based on one choice has different
support from a fraction of 1 based on ten choices. Do not fill unobserved ranges
by extrapolation.

### Features

The primary output is **six history-response profiles per neuron**:

| Spout | Preceding-trial windows |
| --- | --- |
| Left | 5, 10, 20 |
| Right | 5, 10, 20 |

For every profile point, retain its reward-fraction range, mean firing in Hz,
exposure, spike count, contributing trial count, distribution of choice
denominators, and uncertainty. No separate history-correlation scalar is
required. Retain all three windows rather than choosing a winner by default.

### Validation

1. Show coverage of reward fractions and choice denominators for every window.
   Identify profiles driven by one-choice histories or a short session period.
2. Overlapping history windows create dependence. Resample contiguous stretches
   of paired history and firing observations within supported epochs, with
   lengths that account for at least the longest 20-trial history and any
   longer observed dependence. Retain original history context; do not compute
   artificial histories across resampling joins.
3. Compare profiles across session epochs, using the same fraction ranges and
   showing which ranges are supported in each epoch. Changes remain part of
   the profile rather than automatic evidence of failure.
4. Compare 5-, 10-, and 20-trial profiles on common eligible observations as a
   sensitivity check, so differences are not explained solely by which trials
   have a defined history. Show both primary full-support and common-support
   views.
5. Examine sensitivity to choice denominator and fraction-bin definitions.
   Report any restricted analysis separately; do not replace unsupported
   fractions with zero.

If we later ask which window predicts firing best, that is a separate model
comparison. Use optional contiguous validation blocks with history context and
boundary guards handled explicitly; window selection and final performance
assessment require separate roles for the data. Without sufficient data for
that comparison, report all windows without a claim that one predicts best.
Training fit alone is not evidence for the best history timescale.

These profiles describe recent reward experience, not optimal performance or a
learned value estimate. They can overlap with programmed probability, movement,
and current trial responses.

## Settings to fix before implementation

| Setting | Decision |
| --- | --- |
| Probability blocks | Separate native left and right blocks; whole-block firing. |
| Probability-level summaries | Equal-block weighting primary; duration weighting as sensitivity. |
| Trial model | Intercept plus common, side, outcome, and interaction; no slow drift initially. |
| Temporal basis count | Five for each of the four event terms. |
| Reward-history windows | Previous 5, 10, and 20 task trials; side-specific choice denominators. |
| Temporal validation view | Early/middle/late task-time thirds, with coverage checks. |
| Movement bin width, behavioral bins, and lag range | Set before inspecting per-neuron effects. |
| Event support, basis shape, count-bin width, and summary window | Specify together before fitting; the existing 20-ms bins and -1 to +2 s support are available starting choices, not yet fixed for this analysis. |
| Trial-model regularization | Specify a consistent rule and modest sensitivity range before fitting. |
| History-fraction bins and support criteria | Fix before inspecting neuron profiles; keep denominators visible. |
| Resampling length, repetitions, and support thresholds | Specify by axis after inspecting temporal dependence and data coverage; report sensitivity. |

No fixed significance threshold or binary tuning classification is part of the
initial fingerprint. If formal hypothesis tests are added, define the null and
the family of neurons, levels, lags, and windows being tested, and account for
that multiplicity. A surrogate procedure must preserve the structure relevant
to its null; global label shuffling is not a general solution to nonstationarity.

Keep plot text concise and show observed data alongside estimates. All fitting,
resampling, and substantial data processing must run in an existing compute
allocation. Do not perform those operations on a login node or request a new
compute node as part of this workflow.
