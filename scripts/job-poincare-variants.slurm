#!/bin/bash
#SBATCH --job-name=qh-poinc-var   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks
#SBATCH --cpus-per-task=32       # cpu-cores per task
#SBATCH --mem=64G                # memory per node
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load anaconda3/2024.6
conda activate desc-env

declare -A names=(
    [tight-solve]=wqs3-war3-wvol0-wb1-tight-solve
    [no-ess]=wqs3-war3-wvol0-wb1-no-ess
    [maxiter200]=wqs3-war3-wvol0-wb1-maxiter200
    [wqs4]=wqs4-war3-wvol0-wb1
)
for v in tight-solve no-ess maxiter200 wqs4; do
    python -u poincare-qh-optimization-$v.py > outs/poincare-qh-optimization-$v.out 2>&1
    python -u plot-qh-opt-results.py ${names[$v]} > outs/plot-qh-opt-results-$v.out 2>&1
done

python -u lcfs-unnorm-qs-precise_QH.py > outs/lcfs-unnorm-qs-precise_QH.out 2>&1
python -u same-res-eq-solves-QH-cases.py > outs/same-res-eq-solves-QH-cases.out 2>&1
