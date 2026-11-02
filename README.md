# ANCHOR: anchored consensus heads with offsets per reader

Code for **ANCHOR: From Centre Grading Conventions to a Common Scale in Prostate Biopsies**
(ISBI 2027, under review).

Giovanni Casari, Camilla Marella, Moira Ragazzi, Samanta Sacchi, Maurizio Colecchia, Nazario Tenace, Daniele Ravanelli,
Sebastiano Cinalli, Alberto Cavazza, Mauro Iori, Federico Bolelli, Costantino Grana

University of Modena and Reggio Emilia · Azienda USL–IRCCS di Reggio Emilia · IRCCS San Raffaele Scientific Institute,
Milan · Santa Chiara Hospital, ASUIT, Trento

Pathologists at different centres grade the same biopsy with systematic, threshold-specific conventions. When each
centre grades only its own cases, as in federated learning, a centre's convention is confounded with its case mix, so
a per-centre head can neither learn it nor remove it. ANCHOR separates the two with a few **anchors**: biopsies graded
by every centre.

- **Model.** A gated-attention MIL network on frozen UNI2-h patch embeddings gives one case score `s(x)` per biopsy. A
  cumulative-link head gives `P(y >= k | x) = sigmoid(s(x) - theta_k - d_k)`. Here `d = 0` is the *consensus head*
  and `d = delta_c` is the head of centre `c`, with `sum_c delta_c = 0`.
- **Phase 1 (anchors).** The network, `theta` and `delta` are fit on the anchors' grades from every centre. When an AI
  reader is available, its offset profile `omega` is fit on the anchors too. `delta` and `omega` then stay fixed.
- **Phase 2 (local data).**
  - **ANCHOR-Fed:** FedAvg on each centre's own grades, then a refit of `theta`, `delta` and a score scale on the anchors.
  - **ANCHOR-Loc:** each centre fine-tunes alone, adding the AI reader and the anchors' median grade on the consensus head. An optional output ensemble needs one exchange of models.
  - **ANCHOR-Full:** Fed with the AI reader, then the refit.

## Repository

| Path | What it does |
|---|---|
| `anchor/models.py` | Gated-attention MIL, cumulative-link head, multi-reader head (sum-zero, fixed and free offsets), FL centre model |
| `anchor/train_central.py` | Centralised multi-reader training: phase 1, Crossed, Crossed + AI, MedianLabels |
| `anchor/ai_offsets.py` | AI-reader offsets fitted on the anchors after phase 1 |
| `anchor/fl/client.py`, `anchor/fl/job.py` | NVIDIA FLARE client and FedAvg simulator job: BaseFed, FedRep-type, BaseFed + AI, ANCHOR-Fed/-Full, ANCHOR-Loc (single-centre job), GLC |
| `anchor/fl/export.py` | Test predictions of every head for FL / local runs, with and without the anchor refit |
| `anchor/heads.py` | Maximum-likelihood fits on frozen scores (AI offsets, anchor refit) |
| `anchor/train_tracereg.py` | TraceReg competitor (crowd layer + trace regulariser), with or without anchors |
| `anchor/metrics.py`, `anchor/evaluate.py` | QWK / MAE of consensus and own heads on carcinoma biopsies, paired patient bootstrap, Table 1 |
| `datasets/par/` | PAR cohort: download, UNI2-h feature extraction, tissue filter, three-centre study and manifests |
| `configs/par/`, `scripts/run_par.sh` | Hyper-parameters and the full PAR experiment for one seed |

## Installation

```bash
uv venv && uv pip install -e ".[features]"   # or: pip install -e ".[features]"
```

Python 3.12, PyTorch and NVFlare 2.7.2. libvips with openslide is needed only for feature extraction.

## Reproducing the PAR experiments

The public [PAR cohort](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD2323) was scanned on three
scanners and graded by pathologists and AI models. We split its patients into three pseudo-centres, each with its own
reader and its own scanner:

| Centre | Reader | Scanner |
|---|---|---|
| `r1` | pathologist 1 | Leica |
| `r2` | pathologist 2 | Hamamatsu |
| `vir` | Virchow2-based AI model | Grundium |

A task-specific AI model is the AI reader common to every centre. The consensus is the median of the three readers'
grades and is used only for evaluation.

```bash
# 1. data (~1 TB of slides) and UNI2-h features (weights: huggingface.co/MahmoodLab/UNI2-h)
bash datasets/par/download.sh $PAR
python datasets/par/extract_features.py --slides-dir $PAR/component_C_leica_svs --pattern '*.svs' \
    --patch-size 512 --white-threshold 220 --weights $UNI/pytorch_model.bin --output-dir $FEAT/leica
python datasets/par/extract_features.py --slides-dir $PAR/component_B_hamamatsu_ndpi --pattern '*.ndpi' \
    --patch-size 593 --white-threshold 200 --weights $UNI/pytorch_model.bin --output-dir $FEAT/hamamatsu
python datasets/par/extract_features.py --slides-dir $PAR/component_A_grundium_svs --pattern '*.svs' \
    --patch-size 537 --white-threshold 220 --weights $UNI/pytorch_model.bin --output-dir $FEAT/grundium
python datasets/par/tissue_filter.py --slides-dir $PAR/component_B_hamamatsu_ndpi --ext ndpi --patch-size 593 \
    --src $FEAT/hamamatsu --dst $FEAT/hamamatsu_tf
python datasets/par/tissue_filter.py --slides-dir $PAR/component_A_grundium_svs --ext svs --patch-size 537 \
    --src $FEAT/grundium --dst $FEAT/grundium_tf

# 2. pseudo-centres, 5 patient-grouped folds, anchor panels (N = 34, grade-uniform), manifests
python datasets/par/build_study.py --labels $PAR/annotations/slide_labels.tsv --ai-grades $PAR/ai_grades.csv \
    --features $FEAT --out $MAN

# 3. every Table 1 row, seeds 7, 8, 9 (one GPU each), then the table
for s in 7 8 9; do FEATURES=$FEAT/par3c MANIFESTS=$MAN RUNS=runs/par bash scripts/run_par.sh $s; done
python -m anchor.evaluate --table configs/par/table1.yaml --slides $MAN/slides.csv --runs runs/par --out results/par_table1
```

Run the commands from the repository root (the FL job refers to `anchor/fl/client.py`). Patch sizes cover the
same tissue area on every scanner: 512 px at 0.2626 µm/px.

**Training protocol.**
- Adam with learning rate 5e-4, weight decay 1e-4 and offset learning rate 1e-2; dropout 0.5.
- Batches of 32 biopsies, each subsampled to at most 512 patches during training.
- Phase 1: up to 105 epochs with validation early stopping. Everything else: 30 epochs, or 30 FedAvg rounds of one local epoch, keeping the last model.
- Refit by maximum likelihood with an L2 penalty of 0.01 on the offsets.

## Using your own data

1. Save one `npz` per biopsy (bag), with an array `features` of shape (patches, dim).
2. Write manifests with `bag_id`, `patient_id` and one `label_<reader>` column per reader. Grades are 0 (benign) to 5 (ISUP grade group 5); -1 means not graded. Follow the layout of `datasets/par/build_study.py`: `merged/<set>/fold_XX/` for central runs, `<centre>_client/<set>/fold_XX/` for FL clients.
3. Bag ids may start with the centre's code (for example `AB_...`), which is how the FL clients' sets are split.
4. The private multi-centre dataset of the paper is not distributed.

## Citation

```bibtex
@inproceedings{casari2027anchor,
  title     = {{ANCHOR}: From Centre Grading Conventions to a Common Scale in Prostate Biopsies},
  author    = {Casari, Giovanni and Marella, Camilla and Ragazzi, Moira and Sacchi, Samanta and Colecchia, Maurizio and
               Tenace, Nazario and Ravanelli, Daniele and Cinalli, Sebastiano and Cavazza, Alberto and Iori, Mauro and
               Bolelli, Federico and Grana, Costantino},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2027},
  note      = {Under review}
}
```

## License

MIT (see `LICENSE`). PAR data and the UNI2-h weights come with their own terms.

PAR cohort: Muhammad Ali et al., *Virchows Archiv* 2026 (doi:10.1007/s00428-026-04742-5). TraceReg loss adapted from
[maml](https://github.com/ies-research/multi-annotator-machine-learning) (BSD-3).
