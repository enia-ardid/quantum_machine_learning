# Methods

## Data

The task is maze-arm classification from event-aligned PFC and HPC activity.
The notebook contains four experiments: three rewarded arms and all eight arms,
each using event (`qnn_data.pkl`) and pre-event (`qnn_data_pre.pkl`) recordings.
The data are private and are not distributed with this repository.

The upstream [neural_data_analysis pipeline](https://github.com/enia-ardid/neural_data_analysis)
constructs event tensors in
[`construct_tensors.ipynb`](https://github.com/enia-ardid/neural_data_analysis/blob/main/construct_tensors.ipynb).
Each export is a trusted pickle containing these entries:

| Key | Contents |
| --- | --- |
| `tensor_pfc_{subset}` | PFC activity, shaped `(neurons, time bins, events)` |
| `tensor_hpc_{subset}` | HPC activity, shaped `(neurons, time bins, events)` |
| `event_arm_ids_{subset}` | One arm ID per event |
| `trial_ids_{subset}` | One trial ID per event |

`{subset}` is `rewarded` or `all`. Pre-event keys have a `pre_` prefix, such as
`pre_tensor_pfc_rewarded`. The two areas must have matching event counts and order;
neuron counts and time dimensions may differ. For an export stored as
`(time bins, neurons, events)`, pass `tensor_order="time_neurons_events"` to
`load_events`. Confirm the axis convention of the pre-event file before use.
Only load trusted pickle files, since the format can execute code.

Labels are mapped to consecutive integers; arm IDs are retained for plot axes.
The saved three-arm experiments use IDs 2, 4 and 8, with 133 event observations
and 132 pre-event observations. Acquisition metadata, exact time windows and
trained model states are not included.

## Preprocessing and models

Each training fold estimates a mean and standard deviation per neuron across
events and time, adding `1e-9` to the standard deviation. Events are flattened,
projected with PCA separately for each area, concatenated and standardized.
Test events use the fitted training transforms. Requested component counts must
fit the fold's sample and feature dimensions.

All classical models use balanced class weights. Logistic regression uses
`max_iter=1000`; linear and RBF SVC use `C=1` and `tol=1e-3`, with RBF
`gamma="auto"`. Random forests use 100 trees for rewarded arms and 200 for all
arms. Gaussian jitter with standard deviation `1e-5` is added to their train
and test features.

The QNN clamps features to `[-pi, pi]`, multiplies by a learned scalar, and applies
X and Z angle embeddings once. Each variational layer applies trainable RY/RZ
rotations, followed by ring CNOTs or CNOTs over all pairs `i < j`. Pauli-Z
expectations feed LayerNorm, dropout (`0.1`) and a linear classification head.
The circuit uses an analytic state-vector simulator without shot sampling or
hardware noise.

| Task/input | PCs PFC/HPC | QNN topology / layers |
| --- | --- | --- |
| Rewarded / PFC | 5 / 0 | ring / 6 |
| Rewarded / HPC | 0 / 5 | ring / 6 |
| Rewarded / combined | 4 / 1 | ring / 6 |
| Rewarded / extra classical comparison | 30 / 0 | — |
| All arms / PFC | 8 / 0 | all pairs / 8 |
| All arms / HPC | 0 / 8 | ring / 6 |
| All arms / combined | 7 / 1 | ring / 6 |

Rewarded-arm QNNs use AdamW (`lr=0.003`, `weight_decay=1e-4`) and cosine annealing
with `T_max=200`, for up to 600 epochs. All-arm QNNs use Adam (`lr=0.01`) and
ReduceLROnPlateau (factor `0.5`, patience 50), for up to 800 epochs. Both use
full-batch cross-entropy and gradient clipping at 1.

Early stopping uses training accuracy with patience 40 and a `1e-4` improvement
threshold. Accuracy is measured with dropout before the parameter update; the
checkpoint is saved after the update. There is no validation split or nested
hyperparameter search. The dependency versions are pinned, and evaluation
functions accept an explicit random seed.

## Evaluation

The default is three folds. Rewarded-arm classical models use shuffled stratified
event folds. QNNs and all-arm classical models use
[GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html),
keeping events from the same trial together. These splits do not evaluate
generalization across animals, sessions or future time periods.

Folds with fewer than 15 training events or 5 test events are skipped. QNNs and
rewarded-arm baselines require every task class in training; all-arm baselines
require at least two classes. Reports include `n_folds` and `n_evaluated`, since
these rules can produce different test coverage across models. A run with no
valid folds raises an error.

Accuracy and macro-F1 are unweighted fold means, with population standard
deviations (`ddof=0`). Macro-F1 uses the union of labels in each fold's truth and
predictions. Confusion matrices pool held-out predictions, so pooled accuracy
can differ from mean fold accuracy. The `1/K` reference line represents uniform
random guessing, not the majority-class baseline.

The shuffled-label control permutes training labels only. The paired-event
bootstrap uses 5,000 draws, seed 0 and a nominal 95% percentile interval. It
resamples fitted held-out predictions and does not account for within-trial
dependence or variability from retraining. Its output columns are
`event_bootstrap_low/high`.

The optional `permutation_test_baseline` compares mean accuracy over fixed CV
folds with the same statistic for each permuted label array. Preprocessing and
jitter remain fixed. Labels are shuffled within trials when groups are supplied,
or across events otherwise, assuming exchangeability at the selected level.
The tail probability is `(count(null >= observed) + 1) / (B + 1)`, following the
[complete-CV permutation-test convention](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html).
This test runs separately from `run_benchmark`.

## Saved results

The 28 figures and accompanying tables come from the experiments saved in
[`qnn.ipynb` at commit `1e9c0ee`](https://github.com/enia-ardid/quantum_machine_learning/blob/1e9c0ee/qnn.ipynb).
Each experiment includes six confusion matrices and one accuracy comparison.
The README figure is the eight-arm event comparison. These results have not
been regenerated from the current modules.

The saved three-arm comparison uses different splits for classical models and
the QNN, so its bars are not a matched benchmark. The eight-arm models share a
trial-grouped split strategy, but the saved outputs do not report per-model fold
coverage. Neither comparison establishes a quantum advantage.

Several details affect interpretation of the saved tables:

- The saved loader swaps neuron and time axes, whereas the upstream event
  exporter writes neuron-first tensors. That difference changes the normalization
  axis. The exact private files are unavailable, so its effect on the figures
  cannot be checked; current loading follows the upstream convention.
- The three-arm QNN rows say “3 layers”, but the saved function defaults to six.
  The circuit depth used for those figures cannot be established from the outputs.
- Classical `CI_ACC_low/high` entries contain mean shuffled-label accuracy rather
  than confidence limits. The saved `Perm_p` calculation mixes fold scores with
  a mean-CV score and does not provide a valid significance test.

Recomputing these results requires the private exports, confirmed tensor axes
and recorded model settings. The tests exercise the software on synthetic inputs;
they do not reproduce the neural-data experiments.
