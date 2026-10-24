# Evaluation of Multilingual Pretraining for Low-Resource NMT



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



**Locally.**
```bash
git clone https://github.com/YOUR-USERNAME/low-resource-nmt-opus100.git
cd low-resource-nmt-opus100
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# install a CUDA build of PyTorch first if you have a GPU: https://pytorch.org/get-started/locally/
jupyter lab low_resource_nmt_experiments.ipynb
```


## Outputs

Written to `./results/` (and `./data/` for raw splits and tokenizer files):

- Tables: `table3_bleu_results.csv`, `table3b_chrf_results.csv`, `table6_significance.csv`,
  `table_bootstrap_significance.csv`, `table_seed_mean_std.csv`, `table4_ablation.csv`,
  `table_data_quality_audit.csv`, `table_langid_audit.csv`, `table_filtering_before_after.csv`, and others
- Figures: `fig*.png` (BLEU bars/heatmap, training curves, efficiency, error proxy, radar, dashboard, ...)
- `raw_results.json`, `raw_results_checkpoint.jsonl`, `hypotheses_sample.json`


## Data and licences

- Data: OPUS-100 (Zhang et al., 2020), downloaded automatically via Hugging Face `datasets`.
  OPUS-100 is built from OPUS corpora; see the [dataset card](https://huggingface.co/datasets/Helsinki-NLP/opus-100)
  for terms of use. No data is redistributed in this repository.
- Pretrained models: `bert-base-multilingual-cased` and `google/mt5-small` from Hugging Face, under their own licences.
- Code in this repository: MIT (see `LICENSE`).


