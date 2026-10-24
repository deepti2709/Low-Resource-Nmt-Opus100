# Reproducible Evaluation of Multilingual Pretraining for Low-Resource NMT

Code companion to the paper *"Reproducible Evaluation of Multilingual Pretraining for Low-Resource
Neural Machine Translation"* (D. Chopra, M. Lamba, S. Pandey, S. Maheshwari).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/low-resource-nmt-opus100/blob/main/low_resource_nmt_experiments.ipynb)

The notebook trains and evaluates translation systems for **English→Yoruba (En–Yo)** and
**English→Khmer (En–Km)** on real parallel data from [OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100),
and includes a **corpus audit** (train–test duplicates, target-language check, short segments).

## What is in the notebook

| Step | What it does |
|---|---|
| Data | Loads OPUS-100 (`en-yo`, `en-km`), normalises text (NFKC, lowercase), draws fixed-seed splits |
| Audit | Exact and near-duplicate train–test overlap (character-trigram Jaccard ≥ 0.7), target-script / copy checks, short-segment rates |
| Tokenisation | Shared SentencePiece model per pair (BPE; Unigram is used automatically for scripts without spaces, such as Khmer) |
| Systems | LSTM + Bahdanau attention · Transformer (from scratch) · Transformer with mBERT-initialised encoder · fine-tuned `google/mt5-small` |
| Evaluation | BLEU (sacreBLEU) and chrF++, mean ± std over seeds, paired t-test, paired bootstrap, tokenisation ablation, contamination before/after filtering |
| Outputs | CSV tables, PNG figures, raw hypotheses (see below) |

**Not included:** the zero-shot NLLB-200 evaluation reported in the paper is not part of this notebook.

## Quick start

**Google Colab (easiest).** Click the badge above, choose `Runtime → Change runtime type → GPU (T4 or better)`,
then `Runtime → Run all`.

**Locally.**
```bash
git clone https://github.com/YOUR-USERNAME/low-resource-nmt-opus100.git
cd low-resource-nmt-opus100
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# install a CUDA build of PyTorch first if you have a GPU: https://pytorch.org/get-started/locally/
jupyter lab low_resource_nmt_experiments.ipynb
```
A GPU is strongly recommended. Without one the notebook switches to a tiny CPU mode (500/100/100 pairs,
1 seed), which is only a smoke test.

## Configuration (cell "2. Experiment configuration")

| Setting | Default (GPU) | `FULL_SCALE = True` |
|---|---|---|
| Train / val / test pairs per language pair | 3,000 / 400 / 400 | 8,000 / 1,000 / 1,000 |
| Max training steps | 3,000 | 100,000 |
| Evaluate every | 300 steps | 2,000 steps |
| Seeds | 13, 42 | 13, 42, 123, 777, 2024 |
| Subword vocabulary (`BPE_VOCAB`) | 8,000 | 32,000 |

The default run takes roughly 1–2 hours on a free Colab T4. `FULL_SCALE` needs many GPU-hours.
Set `USE_DRIVE = True` in the config cell to save data and results to Google Drive on Colab.

## Outputs

Written to `./results/` (and `./data/` for raw splits and tokenizer files):

- Tables: `table3_bleu_results.csv`, `table3b_chrf_results.csv`, `table6_significance.csv`,
  `table_bootstrap_significance.csv`, `table_seed_mean_std.csv`, `table4_ablation.csv`,
  `table_data_quality_audit.csv`, `table_langid_audit.csv`, `table_filtering_before_after.csv`, and others
- Figures: `fig*.png` (BLEU bars/heatmap, training curves, efficiency, error proxy, radar, dashboard, ...)
- `raw_results.json`, `raw_results_checkpoint.jsonl`, `hypotheses_sample.json`

## Important notes on interpreting results

- **The En–Yo test split of OPUS-100 overlaps heavily with its training split.** In our run, 84.5% of the
  400 sampled test pairs also appear in the training sample. BLEU on this pair is therefore inflated by
  memorisation; use the before/after-filtering cell and the audit tables rather than the raw En–Yo BLEU.
- With the default 2 seeds, standard deviations and significance tests have very little power. Use
  `FULL_SCALE = True` or add seeds before drawing conclusions.
- Tokenisation is not identical across systems: mBERT and mT5 use their own pretrained vocabularies,
  while the from-scratch models share the SentencePiece model trained in the notebook.
- Exact numbers can vary with GPU type, library versions and non-deterministic CUDA kernels, even with fixed seeds.
- The "error analysis" cell is an automated proxy (OOV rate, length ratio, edit distance), not human annotation.

## Data and licences

- Data: OPUS-100 (Zhang et al., 2020), downloaded automatically via Hugging Face `datasets`.
  OPUS-100 is built from OPUS corpora; see the [dataset card](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
  for terms of use. No data is redistributed in this repository.
- Pretrained models: `bert-base-multilingual-cased` and `google/mt5-small` from Hugging Face, under their own licences.
- Code in this repository: MIT (see `LICENSE`).

## Citation

```bibtex
@misc{chopra2026reproducible,
  title  = {Reproducible Evaluation of Multilingual Pretraining for Low-Resource Neural Machine Translation},
  author = {Chopra, Deepti and Lamba, Monika and Pandey, Sakshi and Maheshwari, Shikha},
  year   = {2026},
  note   = {Code: https://github.com/YOUR-USERNAME/low-resource-nmt-opus100}
}
```
Replace this entry with the published paper reference, and add the Zenodo DOI once you have one.

## Contact

Shikha Maheshwari (corresponding author) — shikha.maheshwari@jaipur.manipal.edu
