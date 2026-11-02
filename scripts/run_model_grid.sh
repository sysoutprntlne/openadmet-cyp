#!/bin/bash -l
# Train the 3 x 2 grid of Chemprop models: three training-data sets, each from
# scratch and initialised from CheMeleon.
#
#   admet   challenge + the ADMET/metabolism sets
#   aux     challenge + the auxiliary CYP sets (PubChem, ChEMBL, BindingDB, plus
#           the challenge's own TDI and single-concentration assays)
#   both    challenge + both of the above
#
# Everything else is held fixed: single stage, MAE loss, chemprop defaults, the
# pinned split, and the same seeds, so differences are down to the data and the
# initialisation. The challenge direct-inhibition heads are always trained.
#
# Usage: scripts/run_model_grid.sh [extra train-chemprop args...]
# Environment:
#   SEEDS     space-separated PyTorch seeds (default "0 1 2")
#   OUT_DIR   output root (default results/chemprop/grid)
#   ADMET     override the ADMET dataset names
#   AUX       override the auxiliary dataset names
#   ONLY      run only these model names, e.g. ONLY="both_scratch both_chemeleon"
#   DRY_RUN=1 print the commands without running them
#
# Each run writes <OUT_DIR>/<name>/seed<k>/ and is skipped if its model exists,
# so a killed job can simply be resubmitted.

set -euo pipefail

SEEDS="${SEEDS:-0 1 2}"
OUT_DIR="${OUT_DIR:-results/chemprop/grid}"
ADMET="${ADMET:-tdc_hlm tdc_rlm expansionrx tdc_clearance_microsome_az tdc_half_life_obach tdc_cyp2c9_substrate tdc_cyp2d6_substrate tdc_cyp3a4_substrate}"
AUX="${AUX:-pubchem chembl bindingdb tdi single_conc}"
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-0}"

if [ "$DRY_RUN" != "1" ]; then
    if [ ! -f data/train.csv ] || [ ! -d data/metabolism ]; then
        echo "Missing data/. Run download-data first." >&2
        exit 1
    fi
    # CheMeleon is fetched on first use; compute nodes may have no internet.
    if [ ! -f "$HOME/.chemprop/chemeleon_mp.pt" ]; then
        echo "warning: ~/.chemprop/chemeleon_mp.pt not found; the CheMeleon runs download it" >&2
        echo "         (do that once on a login node if compute nodes are offline)" >&2
    fi
fi

declare -A DATASETS=(
    [admet]="$ADMET"
    [aux]="$AUX"
    [both]="$ADMET $AUX"
)

failed=()
for init in scratch chemeleon; do
    for data in admet aux both; do
        name="${data}_${init}"
        if [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]]; then
            continue
        fi
        init_args=()
        if [ "$init" = "chemeleon" ]; then
            init_args=(--from-foundation chemeleon)
        fi
        for seed in $SEEDS; do
            out="$OUT_DIR/$name/seed$seed"
            if compgen -G "$out/model_*/best.pt" > /dev/null; then
                echo "== $name seed $seed: done already, skipping"
                continue
            fi
            echo "== $name seed $seed"
            cmd=(train-chemprop
                --datasets ${DATASETS[$data]}
                --loss-function mae
                --metrics mae rmse
                --show-individual-scores
                --seed "$seed"
                --output-dir "$out"
                "${init_args[@]}"
                "$@")
            if [ "$DRY_RUN" = "1" ]; then
                printf '%q ' "${cmd[@]}"; echo
            elif ! "${cmd[@]}"; then
                echo "FAILED: $name seed $seed" >&2
                failed+=("$name/seed$seed")
            fi
        done
    done
done

if [ "${#failed[@]}" -gt 0 ]; then
    echo "Failed runs: ${failed[*]}" >&2
    exit 1
fi
