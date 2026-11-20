#!/bin/bash
#SBATCH --job-name=qh-opts        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1               # total number of tasks
#SBATCH --cpus-per-task=32        # cpu-cores per task
#SBATCH --mem=64G         # memory per node
#SBATCH --time=06:30:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load anaconda3/2024.6
conda activate desc-env

# python -u lcfs-normalized-qs-precise_QH.py > outs/lcfs-normalized-qs-precise_QH.out 2>&1
# python -u lcfs-normalized-qs-precise_QH_same_res.py > outs/lcfs-normalized-qs-precise_QH_same_res.out 2>&1
python -u lcfs-unnorm-qs-precise_QH.py > outs/lcfs-unnorm-qs-precise_QH.out 2>&1
# python -u poincare-qh-optimization.py > outs/poincare-qh-optimization.out 2>&1
python -u same-res-eq-solves-QH-cases.py > outs/same-res-eq-solves-QH-cases.out 2>&1
