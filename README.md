# MechineLearningProject

1) What this codebase does

This codebase implements streaming (causal) prediction for code-switching using the SwitchLingua dataset.

Given only the prefix up to position t, the model predicts:

switch_next[t]: whether the next language token will switch languages (binary)

duration3[t]: the length category of the upcoming language segment (3-class), evaluated at switch points

Causality guarantee: training and inference never consume tokens beyond position t.

2) Quickstart commands

All commands should be run from the project root.

(A) Build weakly-supervised dataset (one-time)
PYTHONPATH=. python src/build_weak_dataset.py
PYTHONPATH=. python src/build_weak_labels.py

(B) Baselines (for reporting)
PYTHONPATH=. python src/baseline_eval.py

(C) Train streaming model (prefix-only)
PYTHONPATH=. python src/train_streaming.py

(D) Evaluate on natural distribution (threshold sweep)
PYTHONPATH=. python src/eval_streaming.py

(E) Live demo (token-by-token streaming)
PYTHONPATH=. python src/demo_streaming.py

3) Core design choices
3.1 Skip-other switch label definition

Naive token-adjacent switch definition is fragile due to punctuation tokens labeled as other.
We therefore define switch with skip-other:

current_lang: the most recent non-other language at or before t

next_lang: the first non-other language after t

switch_next[t] = 1 iff current_lang != next_lang

This restores a realistic switch rate (≈0.16 for English–Chinese) and avoids discarding switch signals near punctuation.

3.2 Streaming (causal) setup

For each training example, we sample an index t and form:

Input: prefix tokens [0..t]

Targets: switch_next[t] and duration3[t]

Loss is computed only at the final prefix position (no future tokens visible).

3.3 Class imbalance handling

Switch events are a minority class. We use:

switch point over-sampling during training (p_switch)

class-weighted cross-entropy for switch prediction

Evaluation uses natural sampling (no over-sampling) + threshold sweep.

4) File-by-file guide (src/)
Training / evaluation / demo entrypoints

train_streaming.py
Main streaming training script (prefix-only). Loads StreamingPointDataset, trains XLM-R multitask heads, saves checkpoint to runs/xlmr_streaming.pt.

eval_streaming.py
Natural-distribution evaluation. Expands each sample into multiple (sample, t) points, runs inference prefix-only, sweeps thresholds to report best switch F1 and predicted switch rate. Also reports duration accuracy at valid switch points.

demo_streaming.py
Live token-by-token streaming demo. Prints P(switch_next) and (optionally) duration prediction when exceeding a threshold.

baseline_eval.py
Causal baselines:

Never-switch baseline for switch prediction (F1 = 0)

Majority-class baseline for duration classification

pipeline_preview.py
Debugging tool to print tokenization, LID tags, and generated labels for a few examples. Used to validate preprocessing and label logic.

Dataset / labels

build_weak_dataset.py
Builds processed dataset splits for the target language pair (e.g., English/Chinese). Outputs JSONL files under data_processed/.

build_weak_labels.py
Generates weak supervision labels using token LID and the skip-other definition:

switch_next labels

duration3 labels

dataset_cs.py
Dataset loader used by training/eval. Reads processed JSONL, returns:

tokens

switch labels

duration labels
Contains the skip-other label construction logic.

dataset_streaming.py
Streaming dataset wrapper used for training. Samples a position t and returns:

prefix tokenized input

label_pos (position to read logits)

y_sw, y_dur

Tokenization + language ID

tokenize_simple.py
Tokenization utility for mixed Chinese/English text (splits Chinese chars, English words, and punctuation).

token_lid.py
Token-level language ID wrapper using fastText lid.176.bin, mapping tokens into {en, zh, other} plus simple heuristics.

Model

model_mt.py
XLM-R encoder + dual heads:

switch head (2-class)

duration head (3-class)

5) Artifacts / outputs

Model checkpoint:

runs/xlmr_streaming.pt

Processed data:

data_processed/train.jsonl

data_processed/dev.jsonl

6) Reproducibility notes

Use fixed seeds in dataset sampling and training.

Evaluation uses natural sampling (no training oversampling).

Switch threshold is selected via sweep; the best threshold is printed by eval_streaming.py.

7) Minimal deliverable set (for submission)

Recommended code files to submit (12 files):
