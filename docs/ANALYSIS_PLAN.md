# Active recordings: function, cell type, and anatomy

## What we want to understand

We want to find scientifically interesting structure in the active dynamic-foraging recordings. The embeddings and their unsupervised clusters give us one way to organize the neurons. The GLMs describe their task-related responses. Recorded cell-type labels and anatomical positions let us interpret those responses biologically.

A pattern does not need to outperform firing rate, ISI, waveform, or population coupling to be worth studying. Those properties can help describe or explain a group. Our main questions are whether the pattern is coherent, whether it is supported by individual neurons, and whether a similar organization appears in another recording.

The immediate focus is the active dataset. Passive-to-active ChAT identification remains a separate exploratory possibility and is not required for these analyses. Unlabeled cells stay unlabeled.

## 1. Clusters and task responses

### Describe the functional properties of the groups

Keep the existing embedding clusters fixed, without using GLM responses or cell-type labels to redefine them. For each recording and cluster, examine:

- Full GLM kernels for cue, choice, outcome, left and right licking, and movement where available.
- Response direction, timing, duration, and variation across individual neurons.
- Reward-probability and history terms, using their actual model definitions.
- Recorded spikes aligned to the relevant events, where a promising GLM pattern needs a closer look.
- GLM reliability, the number of contributing neurons, and single-unit versus multi-unit composition.

Keep full-recording coefficients and kernels separate from validation information. The full fits describe the response; held-out gains and fold agreement help judge how much confidence to place in it. A large signed kernel and a reliable improvement in prediction answer different questions.

We should be able to identify neurons with a particular functional property, locate them within the clusters and the continuous embedding space, and see whether the property is shared broadly or concentrated in a subset. A cluster average should not hide opposing responses or a few influential cells.

### Test whether the embedding–function relationship transfers

The next analysis focuses on **one GLM feature at a time**. For the go cue, that feature is its full −1 to +2 s temporal kernel, rather than the full collection of GLM predictors or just an averaged coefficient. Learn one or two linear embedding axes associated with variation in this kernel using December 10 neurons. Describe the actual response shapes, individual cells, existing clusters, recorded labels, and anatomy along those axes. The axes are GLM-guided; the underlying embeddings and original clusters remain unsupervised.

Use two-component partial least squares with a fixed one-component comparison. All scaling and axes are learned from source neurons only. Held-out source-neuron predictions distinguish descriptive training organization from generalization. Freeze the source axes and kernel decoder before evaluating December 18. Direct projection is primary; an unlabeled target-mean adjustment is a separately reported sensitivity, not a fitted rotation. No target GLM values select the model or alignment.

After freezing the cue axes, inspect choice, outcome, and licking features along them. Compare nearby neurons in the cue projection and the original embedding space. Within-session neighborhood coherence and source-to-target feature prediction are separate questions. Other coefficients from the same GLM provide supporting functional interpretation, not statistically independent confirmation. Do not require improvement over physiology baselines or regress away potentially interesting biological organization.

One useful validation is to learn the association in recording A, freeze it, and apply it to recording B using embeddings from the same encoder. Then reveal B's independently fitted GLM features and compare them with the predictions.

The target can be a continuous functional property, such as a signed left-versus-right choice response, rather than a binary class. The association can be represented by a simple embedding readout or by source-cluster response profiles. This readout is supervised by GLM features; the original embedding clusters remain unsupervised.

For this test:

1. Define the functional feature and its sign, timing window, and reliability criteria before inspecting target-session prediction results.
2. Use source neurons only to fit the readout, select its regularization, and learn any scaling or dimensionality reduction. Check prediction on held-out source neurons as well.
3. Apply the frozen mapping to the other recording. Target GLM values must not determine the mapping, cluster correspondence, feature selection, or hyperparameters.
4. Inspect the predicted and observed feature for individual target neurons and for target clusters. Cluster numbers are local to a recording and do not establish a correspondence.
5. Preserve the full target kernels to check whether a successful scalar prediction corresponds to a similar response shape. Report failed or weak transfer alongside successful patterns.

The same encoder already gives the recordings a common coordinate system. A new CCA alignment between unpaired neurons is not automatically justified. A predefined, label-free session-centering sensitivity can assess session offsets; it must be reported separately and cannot be chosen because it fits target GLM responses better.

A first transfer within animal 745502 can use December 10 as the source and December 18 as the target. This tests across-recording correspondence within one animal. Both recordings were used in representation training, so it is a held-out GLM-readout test, not validation on an encoder-unseen animal. Other animals provide a later, distinct level of replication.

### Choice and reward prediction error

Left/right choice is available in the current GLM and is a practical first feature. Separate the direction of the fitted choice effect from evidence that choice-side terms reliably improve prediction. Preserve the side-by-outcome interaction so that outcome-dependent side preference is not mistaken for one fixed preference.

A reward-outcome kernel is not automatically a reward prediction error. An RPE analysis needs an explicit expectation estimate: for example, a behaviorally fitted learning model defining the chosen option's expected value on each trial, followed by a signed outcome-minus-expectation variable. Task-programmed reward probabilities provide a different, externally defined expectation and must be named accordingly. This would be a new GLM specification, not a relabeling of the existing outcome coefficient.

Predicting a neuron's GLM-defined choice response from its embedding is also distinct from decoding the mouse's choice on individual trials. The first analysis concerns functional organization across neurons; trial-by-trial behavioral prediction would be a separate analysis.

## 2. Cell types within functional groups

Use recorded labels to interpret the groups after describing their functional responses. Start with recordings that contain enough labeled cells for the comparison.

There are three connected questions:

- **Composition:** Which recorded cell types are represented in each cluster, and how does this compare with the labeled population in that recording?
- **Within-cluster function:** Do labeled D1 and D2 neurons within a cluster share its response, or is the cluster's average driven by one cell type or an unlabeled subset?
- **Within-type diversity:** Does the same recorded cell type have different GLM responses in different clusters?

These distinguish a group enriched for one cell type from a functional group spanning several cell types, and from functional subdivisions within one cell type. All are potentially useful findings. Sparse labeling and multi-unit labels should remain visible; the unlabeled population is not a confirmed negative class.

## 3. Anatomy, clusters, and task responses

Use CCF anterior–posterior, dorsoventral, and mediolateral coordinates, together with recorded region labels, to examine where the functional groups are located.

- Map cluster membership and individual GLM features in anatomical space.
- Compare regions such as CP and ACB where those annotations are present.
- Look for spatial concentrations and continuous response gradients within regions and clusters.
- Where labeling permits, examine whether cell-type-specific responses vary with anatomical position.

Anatomical organization is itself a possible result. We should first describe it rather than automatically remove it as a nuisance variable. Keep CCF coordinates distinct from probe-relative depth, account for which probe/track sampled each location, and do not infer missing anatomy from embeddings. A gradient observed along one recording track should not be presented as a comprehensive map of the striatum.

## How the three directions connect

The first direction establishes **what the neurons do in the task** and how those properties relate to embedding groups. The second asks **which recorded cell types express those properties**, including differences within a group and within a cell type. The third asks **where those functional populations are located**.

The same neuron identifiers connect all three views. An interesting result might be a common task response shared by D1 and D2 neurons within one cluster, a particular response restricted to a subset of D2 neurons, or a response that changes with anatomical position across several clusters. These are examples of questions to investigate, not findings established so far.

Cross-recording transfer provides one way to strengthen the first direction: an embedding neighborhood or mapping identified in one recording predicts a related functional property in another. The cell-type and anatomical views then help determine what has recurred biologically. Recurrence does not require identical cluster counts or matching cluster numbers.

## Available starting material

The existing September 12 cluster–GLM overlap contains:

| Recording | Units | Recorded D1 | Recorded D2 | CCF coordinates |
| --- | ---: | ---: | ---: | --- |
| December 10, 2024 — animal 745502 | 192 | 58 | 51 | Available |
| December 18, 2024 — animal 745502 | 247 | 40 | 0 | Missing |
| March 31, 2025 — animal 754313 | 153 | 3 | 25 | Available |
| April 2, 2025 — animal 754313 | 46 | 0 | 1 | Available |
| April 9, 2025 — animal 754313 | 58 | 3 | 5 | Available |

These are analysis-overlap counts, not all units in the raw recordings. December 10 is the best initial recording for joining function, recorded cell type, and anatomy. March 31 is useful for functional and anatomical follow-up. December 18 supports within-animal functional transfer but lacks CCF annotations and the movement covariate in the current GLM cache.

For December 10 and 18, matched exports and clusters exist for the historical active-pretrained encoder, the newer passive-pretrained encoder, and its active-fine-tuned version. Use each encoder's own shared space; do not mix coordinates from different models. The fine-tuned encoder can be the first analysis, with the other saved encoders reported as sensitivities rather than selected by target-session success.

## First concrete analysis

Start with the signed choice-side response and its full time course in December 10, using the existing clusters and frozen GLM fits. Use an outcome-averaged right-minus-left log-rate contrast as the primary continuous target, with the exact coding checked against the GLM design. Show outcome-conditioned contrasts and choice-side reliability alongside it.

Fit a modest, source-only embedding readout and evaluate it on held-out December 10 neurons, then apply it unchanged to December 18. Also transfer the source clusters' mean response profiles through embedding proximity, retaining an explicit indicator for weakly supported target matches. Compare predictions with December 18's GLM features after freezing the mapping. Record within-session organization and cross-session transfer as separate outcomes.

The first outputs should include per-neuron feature and prediction tables, cluster kernel plots, embedding views colored by observed and predicted choice response, and a concise account of what transfers and what does not. No new encoder training or GLM fitting is needed for this choice-response pilot. Run numerical work only in an existing authorized CPU allocation.

### Initial pilot completed — October 16, 2026

The [choice-response report](results/experiments/active_function/choice_dec10_to_dec18/index.html) contains 192 December 10 neurons and 247 December 18 neurons, with full contrast curves, source-only readouts, individual-neuron reliability information, and serialized predictions. The signed outcome-averaged 0–1 s right-minus-left contrast showed little prediction in the primary fine-tuned/original-space analysis: held-out source-neuron correlation −0.067 and target-session correlation −0.007. Source-cluster response transfer was similarly weak. Thus this particular summary has not provided evidence of transferable functional organization.

This result does not assess every task property or the full temporal structure of each response. Choice-side reliability includes side-by-outcome terms, so evidence that a cell carries choice-related information is distinct from a consistent average right/left preference. Full kernels, outcome-conditioned contrasts, and the individual neurons remain the starting material for biological interpretation; the result is not a reason to discard other cluster–GLM associations.

### Single-feature cue axes completed — October 16, 2026

The [go-cue axis report](results/experiments/active_function/cue_axes_dec10_to_dec18/index.html) uses the same 192 December 10 and 247 December 18 neurons with the fine-tuned encoder. Two PLS axes are learned only from the source go-cue kernel (−1 to +2 s); the existing unsupervised clusters are unchanged. The output preserves full kernels, per-neuron coordinates and metadata, frozen model parameters, held-out predictions, and separate reliability information.

The axes explain 9.8% of full-kernel variation across held-out December 10 neurons (neuron-bootstrap interval 4.5–14.3%). In December 18, predicted and observed mean 0–1 s cue effects correlate at r=0.332 (interval 0.213–0.432), including r=0.294 among 131 single units. Full-curve prediction does not transfer accurately: target kernel R² is −0.044. Unlabeled mean adjustment does not improve it. The source single-unit-only evaluation is less precise (53 units; kernel R² 0.052 with an interval crossing zero). These are conditional neuron-level uncertainty estimates within two encoder-trained recordings of the same animal, not animal-level replication.

There is an interpretable cue–outcome gradient. In each session, high first-axis neurons show stronger peri-cue kernels and more negative early reward-minus-omission kernels. The high versus low first-axis quartiles have median 0–1 s cue effects of +0.110 versus −0.064 in December 10, and +0.108 versus −0.088 in December 18. Their median outcome contrasts are −0.147 versus +0.033 and −0.185 versus +0.035, respectively. Responses begin before the go cue, so this should not be called a purely sensory cue response or an RPE signal.

The high source quartile contains 35 C1 and 13 C2 units, including 19 D1 and 6 D2 labels; the low quartile spans five clusters and includes 16 D1 and 13 D2 labels. Firing rate also varies along this axis (high/low medians about 5.2/0.55 Hz in December 10 and 5.0/0.30 Hz in December 18). These physiological differences are context for interpreting the functional gradient. They are not automatically a reason to discard it. The second source axis also relates to CCF coordinates; December 18 has no CCF data for anatomical transfer.

Outcome was inspected after freezing cue axes. Predicting each December 18 unit’s mean outcome effect from its ten December 10 neighbors in the cue projection gives r=0.366 and R²=0.126; evaluating target single units gives r=0.328. The same prediction using original embedding neighbors gives r=0.475 and R²=0.187 (target single units: r=0.479). Thus the outcome correspondence is present in the original embedding space as well. Average left/right preference and cross-session lick effects show little transfer in the cue projection. Within-target neighborhood coherence is reported separately from source-to-target prediction.

This supports a focused next question: which neurons and existing clusters share the peri-cue/early-outcome association, and does it appear in their actual event-aligned firing? Other coefficients from the same GLM are supporting functional interpretation rather than independent confirmation. Recorded types and anatomy can help explain the gradient after its individual-neuron support is inspected.

### Broader individual-feature transfer completed — October 16, 2026

The [individual-feature report](results/experiments/active_function/multifeature_dec10_dec18/index.html) extends this analysis to 12 GLM terms and four derived choice/outcome contrasts, in both December 10 → 18 and December 18 → 10 directions. Each temporal kernel gets a separate two-component PLS model; each probability/history coefficient gets a one-component model. Source-only fitting, held-out source neurons, frozen target evaluation, and original-embedding neighbor prediction are retained. All features and both directions are reported. Movement cannot be compared because December 18 lacks it; December 11 lacks the required behavior.

The most useful correspondence is the early, 0–1 s reward-minus-omission effect: feature-specific axes predict it across sessions at r=0.448 and r=0.276 in the two directions. Single-unit evaluation gives r=0.456 and r=0.300. Absolute summary prediction is better in the forward direction (R² 0.161) than in reverse (−0.060). Full outcome-kernel transfer is poor because later response structure differs substantially; strong within-recording kernel prediction does not imply an accurate transferred time course.

Left reward-probability effects show modest recurring associations (r=0.218 and 0.290); previous right-versus-left choice effects also recur weakly (r=0.152 and 0.201). Both remain positive in single-unit evaluations. Go-cue and common choice/outcome responses have recurring associations as well. These are programmed-probability and history effects, not inferred value or RPE. Ordinary right/left preference and rewarded-trial side preference do not show positive transfer. The unrewarded-trial side contrast has a weaker association in both directions.

Licking illustrates why full temporal kernels should be preserved. Left-lick full-kernel pattern correlations are 0.340 and 0.169, while the fixed ±0.1 s average does not transfer positively. Full-kernel R² is 0.074 forward and −0.248 in reverse; the reverse association is weaker among single units. Right-lick temporal patterns show smaller recurring associations. No full temporal kernel has positive cross-session R² in both directions.

Exploratory multiplicity correction covers 108 distinct tests jointly: 64 full-feature tests and 44 fixed-window temporal-summary tests, across both directions and both readouts. Scalar summaries are not counted twice. Summary significance was added after inspecting the full-kernel results, without changing the existing summary windows or fitting new axes. Neuron-bootstrap intervals, SUA evaluation, and parent-group reliability subsets are saved. The two directions are not independent biological replicates, and the session-specific movement covariate difference remains a limitation.

### Observed-firing check completed — October 17, 2026

The [observed-firing report](results/experiments/active_function/outcome_observed_dec10_dec18/index.html) tests the interpretation against saved spike counts, using CPU allocation 9094536. Each neuron's score is predicted by the outcome readout trained on the other recording. Target GLM values and observed firing do not choose the groups. High and low quartiles are displays of a continuous score, not new cell classes or replacement clusters.

The analysis includes 192 December 10 units (53 single units) and 247 December 18 units (131 single units). Rewarded and omitted trials are paired within choice side, both programmed reward probabilities, and 20-trial time blocks, giving 127 and 125 pairs. Responses are aligned to outcome registration. The primary contrast is reward minus omission during 0–1 s, minus its pre-event contrast during −1 to −0.2 s. Absolute changes in Hz and changes divided by each unit's session mean rate are both retained. No units are excluded based on physiology.

**The direct reward-specific result is mixed and weaker than the GLM-to-GLM transfer.** Spearman correlations between the frozen score and the matched, baseline-adjusted observed outcome contrast are:

| Evaluation | December 10 | December 18 |
| --- | ---: | ---: |
| All units, Hz | −0.044 | 0.147 |
| All units, relative to session mean | 0.048 | 0.129 |
| Single units, Hz | 0.157 | 0.224 |
| Single units, relative to session mean | 0.374 | 0.187 |

The December 10 single-unit relative effect has a positive trial-block bootstrap interval (0.133–0.508). December 18's relative-effect intervals include zero when resampling trial blocks, for both all units and single units. Interleaved trial subsets do not consistently reproduce the outcome association. Matching also matters: the all-trial December 18 association is near zero. These results do not establish robust reward-selective populations across both sessions. The own-session GLM outcome effect correlates more strongly with observed matched outcome contrasts (all-unit Spearman 0.527 and 0.361 in Hz), so weak embedding transfer does not by itself invalidate the GLM feature.

**A different, exploratory pattern is clearer: common activation versus suppression around choice/outcome.** After inspecting the reward and omission traces, we measured their equal-weight average change from baseline, using exactly the same frozen scores, groups, trial matching, and time windows. Low-score neurons tend to increase firing and high-score neurons tend to decrease firing on both rewarded and omitted trials. The common-response correlations are −0.519 and −0.585 in Hz, and −0.600 and −0.454 relative to session mean firing. Single-unit values remain negative: −0.400 and −0.513 in Hz, and −0.363 and −0.419 on the relative scale. This characterization was added after seeing the traces and is a discovery to test prospectively, not a replacement success criterion for the original outcome-selectivity check.

The common-response relationship has the same negative direction in all three interleaved trial subsets, in both recordings, for both all units and single units, on both scales. Its trial-block bootstrap intervals exclude zero in all eight session/sorting/scale evaluations. This supports consistency within these recordings; it does not remove the post hoc status or establish independent biological replication.

Rewarded trials have more sustained post-outcome licking even after the task matching. Cue, choice, registered outcome, and consumption occur close together. Thus the common pattern may reflect action-related activation/suppression, a cue-to-action transition, consumption, or a mixture. The conditional GLM outcome kernel and a marginal event-aligned firing contrast measure different things. These are the same recordings used to compute the embeddings and GLMs, and both recordings were used in encoder training. Neither the observed check nor the two transfer directions is independent animal-level replication.

## Concrete scientific tests following the observed-firing check

### Working claim and boundaries

The current candidate finding is **a recurring embedding-associated gradient of activation versus suppression around a task transition, with a weaker and less stable reward-specific component**. The strong common-response association is exploratory. We have not established a discrete cell class, reward prediction error, a cholinergic population, or the causal event driving the response.

The useful next steps distinguish biological explanations of this gradient and identify the neurons expressing it. They do not require the embedding to beat firing rate or other physiological baselines. Continue reporting those properties as part of what the populations are like.

### Test 1 — Is the recurring property common task activation/suppression, reward selectivity, or both?

**Prediction.** The frozen other-session outcome score will order the common response more consistently than it orders reward-minus-omission responses. A low score predicts a positive common change and a high score predicts a negative common change, without requiring the high-score group to be reward-excited.

**Fixed quantities.** For each neuron, let R and O be baseline-adjusted 0–1 s firing changes on matched rewarded and omitted trials, respectively. Define common response C = (R + O)/2 and outcome selectivity D = R − O. Average choice sides equally. Use −1 to −0.2 s as baseline, retain full traces, and report both Hz and session-rate-relative scales. The continuous frozen score is primary; fixed quartiles illustrate it. Keep all-unit and single-unit results separate.

**Concrete analysis.** Plot individual neurons in the C–D plane, color by the existing clusters, and show where the frozen score runs through that plane. Preserve trial-block uncertainty and trial-subset consistency for each property. Compare the observed C and D with the individual cue, common choice/outcome, and outcome GLM kernels to establish what each conditional feature describes. Do not compress those kernels into a new whole-model fingerprint. Use existing held-out GLM predictions to check whether the full model reconstructs the observed common response while its conditional outcome term describes only a smaller difference.

**Scientific decisions.** If C recurs but D does not, focus the finding on task-related activation/suppression. If both recur, test whether reward selectivity forms a second functional dimension within the common-response groups. If neither recurs in additional data, restrict the claim to the current recordings. New projections fit to C would be a new analysis and must not be described as validation of the original outcome score.

### Test 2 — What part of the task drives the shared response?

**Competing predictions.** A cue-associated response stays aligned to the cue as reaction time varies. An action-associated response stays aligned to the registered choice. A consumption-associated response changes with licking duration or lick-bout timing. A reward-specific contribution produces a reproducible rewarded-minus-omitted difference beyond the common transition, without requiring a large absolute firing increase.

**Concrete analysis.** Keep the embedding score and clusters fixed. View the same neurons around cue, choice/outcome, and lick-bout onset; compare short- and long-reaction-time trials and short- and long-licking trials. Retain rewarded and omitted trials separately and describe preceding firing. Use natural timing variation to distinguish alignments, with trial-count and timing-overlap checks before interpretation. Start with the observed traces and existing GLM kernels; introduce a targeted alternative GLM only if correlated event terms prevent interpreting their contributions.

**Scientific decisions.** Consistent timing relative to an event supports that functional description. If events are too tightly coupled, retain the combined “peri-choice/outcome” label rather than forcing an attribution. Licking-related organization would itself be a useful result. Matching on post-reward licking changes the question and can remove reward-mediated behavior; it is a descriptive sensitivity, not proof of a direct reward effect or its absence. December 18 lacks the movement covariate, so movement-based comparisons must remain session-specific.

### Test 3 — Which biological populations express this organization?

**Predictions to distinguish.** Existing clusters may correspond to activation versus suppression; alternatively, a continuous response gradient may cut across them. Within a cluster, D1 and D2 neurons may share a response, or they may differ. Within one recorded cell type, different clusters may identify functional subgroups. A common response could also vary with CCF location or region.

**Concrete analysis.** Join C, D, their full traces, and frozen scores to the unchanged unsupervised clusters, recorded labels, CCF coordinates, region, depth, waveform, ISI, and firing rate. First show individual-neuron distributions for every cluster. In December 10, compare D1 and D2 within clusters that contain both labels, and compare each type across clusters. Map the same response measures across CP/ACB and CCF coordinates, retaining probe/track identity where available. Report sparse groups rather than interpreting their means as established populations.

**Scientific decisions.** Similar function across D1 and D2 within a cluster supports a functional grouping spanning recorded types. Differences within the same type across clusters support within-type diversity. A spatial gradient connects function to sampled anatomy and is not automatically a confound to remove. A continuous pattern without cluster boundaries supports an axis rather than distinct classes. Tagged multi-units cannot establish a single neuron's molecular identity; preserve sorting status. December 18 has D1 labels but no D2 labels or CCF coordinates, so it cannot replicate D1-versus-D2 or anatomical findings.

### Replication design shared by these tests

Lock the current score, signs, response windows, matching rules, and primary C/D comparisons before inspecting another recording. Use the same encoder coordinates without a target-GLM-guided rotation. Evaluate each recording separately before pooling, and count recordings and animals as the biological replication units rather than treating hundreds of neurons as hundreds of independent animal replicates. Neuron and trial-block intervals answer different within-recording uncertainty questions; neither alone measures across-animal uncertainty.

The current fine-tuned exports cover the three 745502 recordings; December 11 lacks the behavior needed here. Other animals with GLMs and anatomy, including March 31 in animal 754313, are useful next targets, but require exports from the same encoder for direct frozen-score transfer. An analysis using their historical encoder embeddings is a separately named sensitivity, not the same coordinate-space test. Same-animal recurrence and new-animal recurrence should be reported separately. Freeze one source mapping as primary before evaluating a new target; do not select whichever source works best afterward. Additional recordings can validate the newly specified common-response hypothesis; reusing December 10 and 18 remains characterization of the discovery data.

### Reward expectation is a later, conditional branch

If a reproducible outcome-selective component emerges, test how it varies with chosen-option programmed reward probability. For a signed reward-minus-expectation hypothesis, a positive-coding neuron is predicted to respond more to an unexpected reward and more negatively to an unexpected omission; the directions reverse for negative coding. First establish that the available trials support those comparisons. Programmed probabilities are not the animal's inferred beliefs. To claim an RPE-like association, fit a behavioral learning model without neural data, check its held-out choice predictions, and then test the neural dependence on its expectation and signed prediction error alongside action and licking. This would support an RPE-like functional relationship, not a dopamine measurement or a causal mechanism.

### Immediate priority

The next analysis should map the **common activation/suppression response and the separate outcome contrast onto the fixed clusters**, starting with December 10's individual neurons and labeled D1/D2 units, then examining December 18's functional recurrence. Alongside that, use reaction-time and licking variation to determine what the common response is aligned to. This directly connects function, cell type, and anatomy while avoiding a premature reward/RPE interpretation. The observed-firing check above is completed; these hypothesis-specific follow-ups and additional-session replication remain planned.

## Revised starting point — October 21, 2026

Start by looking at individual GLM effects on **unsupervised UMAP and PCA views of December 10**, before fitting another functional alignment or making cluster-level comparisons. The [embedding maps](results/experiments/active_function/glm_maps_dec10/index.html) use the existing fine-tuned UMAP and PCA fitted to normalized embeddings alone. Every coloring uses identical positions within each projection. The initial effects are cue, common choice/outcome, right-minus-left choice, and reward-minus-omission. Main maps show signed 0–1 s effects; separate pre-event, early, and late maps preserve temporal distinctions. Full kernels and validation information remain available. No separation in PC1/PC2 would rule out only an obvious gradient in that view, not functional structure in the complete embedding.

These maps were completed on CPU allocation 9265562: 210 units in the unchanged session UMAP and the new PCA, with matched full GLM features for 192 units, including 53 single units. PC1 explains 47.0% and PC2 13.6% of normalized embedding variance. The temporal maps show more visible pre/early cue structure than the broad 0–1 s average, while left/right choice effects remain intermixed in these two-dimensional views. This is descriptive visual inspection, not an axis-selection rule or a test of cross-session correspondence. Nine PNGs, reusable coordinate/feature tables, full kernels, and PCA parameters are saved in the report directory.

Two complementary interpretations remain of interest: fixed unsupervised clusters can describe functional groups, and continuous embedding directions can describe variation crossing cluster boundaries. A direction may correspond to several GLM properties together. For example, a cue-associated increase and an outcome-associated decrease may share one direction; we should describe that combination rather than require each axis to isolate one predictor.

### Compare functional directions without requiring identical embedding geometry

The same encoder supplies a common numerical coordinate system, but it need not preserve the same embedding-to-function mapping across recordings. The earlier frozen transfer test was deliberately strict. A second question is whether each recording independently contains a similar **functional organization**, even when its embedding directions differ.

Learn directions separately within each recording using a selected feature, or a small explicitly chosen set of features. Begin with one- or two-component PLS; regularized CCA after an unsupervised dimension reduction is a possible sensitivity. These are GLM-guided directions, even if the preceding PCA is unsupervised. Preserve a feature-by-axis loading table and temporal kernels so that the functional meaning of every direction remains visible. Avoid combining every GLM term into an undifferentiated fingerprint. Use training-neuron scaling for GLM targets if multiple effects are combined, document that weighting, and choose complexity within training data.

Within each recording, test embedding-to-feature associations on neurons omitted from fitting the directions. Separately use the existing held-out-trial GLM results to assess whether the neural response estimates are reliable. Those validation splits answer different questions. If two directions are similarly strong and their basis rotates across fits, compare the small subspace rather than insisting that axis 1 must always match axis 1.

**CCA requires paired observations.** Within a session, each neuron's embedding and its GLM features are naturally paired. Arbitrarily pairing different neurons from different sessions, even after PCA, is not a valid cross-session CCA. Cross-session matching requires explicit shared anchors, such as a small set of functional effects, or tracked neurons if they exist. Matching using GLM effects is function-guided alignment and must be named as such. An embedding-only distribution alignment is also possible but cannot establish a unique functional correspondence from similar cloud shapes alone.

A practical first comparison is to learn a direction in each session and compare its association with a few named GLM features on held-out neurons. Similar response combinations in both sessions would support recurring functional organization without requiring a global rotation of the entire neuron cloud. Only if this is informative should we fit a low-dimensional alignment of the functional subspaces. A rotation changes orientation but cannot correct different scaling or general distortions; more flexible maps need more anchors and stronger validation. One feature can orient a one-dimensional score, but cannot determine a unique rotation of a high-dimensional embedding.

If target-session GLM features help construct an alignment, agreement with those same fitting values is a description of the alignment, not independent evidence that it worked. Evaluate neurons excluded from fitting the entire pipeline, and, where useful, GLM features not used to choose or align the directions. Held-out features provide a separate generalization question but can still be statistically related to the training features. Call this **transfer after target-session calibration**, distinguishing it from the earlier uncalibrated frozen transfer. With only two recordings, keep the conclusion at the level of within-animal recurring organization.

The immediate deliverable is the single-session visualization. Separate-session functional axes, any alignment, and their validation are subsequent analyses; the new plotting command does not perform them.

### Observed population overview before individual trial trajectories

The [December 10 population PSTH](results/experiments/active_function/population_psth_dec10/index.html) was completed on October 21, 2026 using CPU allocation 9265562. It includes all 366 raw non-noise units (118 SUA, 248 MUA), including units without embeddings or fitted GLMs. Recorded annotations include 67 D1 and 57 D2 labels. Region groups run from superficial to deep: MOp6a (42), MOp6b (2), cing (1), ccg (3), CP (202), and ACB (116). Rows are ordered by decreasing stored probe depth within each region, with no response-based sorting: this recording's probe coordinate increases toward cortex, opposite to CCF DV. The display and saved cache row order were corrected together after initially plotting this coordinate in the opposite direction. Region annotations are preserved as recorded, including white-matter annotations.

Original spike timestamps were counted in exact 50-ms bins from −2 to +3 s around the go cue and averaged across 526 valid pre-optotagging trials. The full window had to fit inside each trial. Raw Hz is shown alongside a companion view subtracting each neuron's −1 to −0.2 s baseline and dividing by its mean rate in the displayed window. The relative view makes slower units visible but is not a reliability or significance measure. Trials are pooled at their observed choice/outcome frequencies. Choice occurs a median 130 ms after the cue, so this alignment includes closely spaced choice, outcome, and licking events rather than isolating a sensory cue response.

These pooled views are a starting point for choosing individual units to inspect, not evidence that their tuning is stable. Exact chronological unit-by-trial-by-bin counts and unit identities are cached in the same directory. The next step is selected-neuron chronological heatmaps, with task labels and local response estimates, to assess changes across trials before treating any full-recording GLM feature as a persistent functional description.
