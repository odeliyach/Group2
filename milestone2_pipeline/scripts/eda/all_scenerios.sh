#!/bin/bash
#SBATCH --job-name=download-caml-scenarios
#SBATCH --output=logs/caml_download_%j.out
#SBATCH --error=logs/caml_download_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --partition=studentkillable
#SBATCH --chdir=data/caml/cam_lds/
# Submit from the workshop ROOT directory:  sbatch scripts/eda/all_scenerios.sh

BASE="https://zenodo.org/records/18861762/files"

SCENARIOS=(
    scenario_2_cron
    scenario_2_rootkit
    scenario_3_ssh_apt
    scenario_3_ssh_healthcheck
    scenario_3_ssh_puppet
    scenario_3_vnc_apt
    scenario_3_vnc_healthcheck
    scenario_3_vnc_puppet
    scenario_4
    scenario_5
    scenario_6_macro_binary
    scenario_6_macro_cron
    scenario_6_plugin
    scenario_6_screensharing_binary
    scenario_6_screensharing_cron
    scenario_7
)

echo "============================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $(hostname)"
echo "Start      : $(date)"
echo "Downloading ${#SCENARIOS[@]} scenarios"
echo "============================================"

for scenario in "${SCENARIOS[@]}"; do
    ZIP="${scenario}.zip"
    DEST="${scenario}/"

    if [ -d "$DEST" ]; then
        echo "  [SKIP] $scenario already extracted"
        continue
    fi

    echo "  Downloading $scenario..."
    wget -q --show-progress "${BASE}/${ZIP}" -O "$ZIP"
    if [ $? -ne 0 ]; then
        echo "  [ERROR] Failed to download $scenario"
        continue
    fi

    echo "  Extracting $scenario..."
    unzip -q "$ZIP" -d "$DEST"
    if [ $? -eq 0 ]; then
        rm "$ZIP"
        echo "  [DONE] $scenario"
    else
        echo "  [ERROR] Failed to extract $scenario"
    fi
done

echo "============================================"
echo "Finished   : $(date)"
echo "All scenarios:"
ls -lh | grep "^d"
echo "============================================"
