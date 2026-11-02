#!/bin/bash
# Every PAR row of Table 1 for one seed (paper: seeds 7, 8, 9; anchor panel d = seed - 7). One GPU is enough.
#   FEATURES=<features>/par3c MANIFESTS=<manifests> RUNS=runs/par bash scripts/run_par.sh 7
# Then, once the three seeds are done:
#   python -m anchor.evaluate --table configs/par/table1.yaml --slides $MANIFESTS/slides.csv --runs $RUNS --out results/par_table1
# SMOKE=1: fold 1 only, 1 epoch, 2 FedAvg rounds (checks the whole flow in ~30 min; evaluate with --seeds 7 --allow-partial).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}   # the NVFlare client processes import the anchor package
SEED=${1:?seed}
D=$((SEED - 7)); N=34; A=anchors${N}_d$D
: "${FEATURES:?}" "${MANIFESTS:?}" "${RUNS:=runs/par}"
export MANIFEST_ROOT=$MANIFESTS
M=$MANIFESTS/merged
SC=""; SF=""; ST=""
if [ "${SMOKE:-0}" = 1 ]; then SC="--folds fold_01 --epochs 1"; SF="--folds fold_01 --set num_rounds=2"; ST="--folds fold_01 --epochs 1"; fi
CENTRAL="python -m anchor.train_central --config configs/par/central.yaml --features-root $FEATURES --seed $SEED"
FL="python -m anchor.fl.job --config configs/par/fl_base.yaml --features-root $FEATURES --seed $SEED --skip-existing $SF"
EXPORT="python -m anchor.fl.export --centres r1 r2 vir --features-root $FEATURES --test-set $M/crossed"
P1=$RUNS/phase1_N${N}_s$SEED
AIO=$RUNS/ai_offsets_N${N}_s$SEED

# ---- references (centralised: fully crossed grades) ----
$CENTRAL --folds-dir $M/crossed --sources r1 r2 vir --output-dir $RUNS/crossed_s$SEED --skip-existing $SC
$CENTRAL --folds-dir $M/crossed --sources r1 r2 vir ai --free-sources ai --output-dir $RUNS/crossed_ai_s$SEED --skip-existing $SC
$CENTRAL --folds-dir $M/crossed --sources median --consensus-sources median --output-dir $RUNS/median_labels_s$SEED --skip-existing $SC

# ---- BaseFed, BaseFed + AI reader, FedRep-type (FL, own grades only) ----
$FL --output-dir $RUNS/basefed_s$SEED && $EXPORT --run $RUNS/basefed_s$SEED
$FL --output-dir $RUNS/basefed_ai_s$SEED --set offsets=zero --set second_column=label_ai && $EXPORT --run $RUNS/basefed_ai_s$SEED
$FL --output-dir $RUNS/fedrep_s$SEED --set offsets=trainable && $EXPORT --run $RUNS/fedrep_s$SEED --offsets trainable

# ---- ANCHOR phase 1: network, theta and centre offsets on the anchors (step budget of a 30-epoch run) ----
$CENTRAL --folds-dir $M/$A --sources r1 r2 vir --epochs 105 --patience 35 --select-epoch --output-dir $P1 --skip-existing $SC
python -m anchor.ai_offsets --phase1-dir $P1 --anchor-set $M/$A --features-root $FEATURES --out $AIO

# ---- ANCHOR phase 2 ----
ANCH="--set manifest_set=own_$A --set offsets=fixed --set phase1_dir=$P1"
$FL --output-dir $RUNS/anchor_fed_s$SEED $ANCH
$EXPORT --run $RUNS/anchor_fed_s$SEED --phase1-dir $P1 --anchor-set $M/$A
$FL --output-dir $RUNS/anchor_full_s$SEED $ANCH --set second_column=label_ai --set ai_offsets_dir=$AIO
$EXPORT --run $RUNS/anchor_full_s$SEED --phase1-dir $P1 --anchor-set $M/$A
for C in r1 r2 vir; do  # ANCHOR-Loc: each centre alone, no exchange after phase 1
  $FL --output-dir $RUNS/anchor_loc_${C}_s$SEED $ANCH --set "clients=[${C}_client]" --set second_column=label_second \
      --set ai_offsets_dir=$AIO --set consensus_flag=anchor
  $EXPORT --run $RUNS/anchor_loc_${C}_s$SEED --phase1-dir $P1
  $FL --output-dir $RUNS/anchor_loc_noai_${C}_s$SEED $ANCH --set "clients=[${C}_client]" --set second_column=label_median
  $EXPORT --run $RUNS/anchor_loc_noai_${C}_s$SEED --phase1-dir $P1
done

# ---- competitors ----
$FL --output-dir $RUNS/glc_s$SEED --set manifest_set=own_${A}_glc --set glc=true
$EXPORT --run $RUNS/glc_s$SEED --glc $MANIFESTS --glc-set own_${A}_glc
python -m anchor.train_tracereg --config configs/par/central.yaml --folds-dir $M/nested --features-root $FEATURES \
  --centres r1 r2 vir --seed $SEED --output-dir $RUNS/tracereg_s$SEED $ST
python -m anchor.train_tracereg --config configs/par/central.yaml --folds-dir $M/nested_$A --features-root $FEATURES \
  --centres r1 r2 vir --seed $SEED --output-dir $RUNS/tracereg_anchors_s$SEED $ST
