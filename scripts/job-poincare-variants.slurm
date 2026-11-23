#!/bin/bash
#SBATCH --job-name=qh-poinc-repro
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH -o ./outs/slurm-poincare-repro.out

module purge
module load anaconda3/2024.10
conda activate desc-env

echo "DESC /home/ye2698/DESC"
git -C /home/ye2698/DESC log -1 --decorate=short --pretty="%h %d %s"
git -C /home/ye2698/DESC status --short

python poincare-qh-optimization.py > outs/poincare-qh-optimization.out 2>&1
