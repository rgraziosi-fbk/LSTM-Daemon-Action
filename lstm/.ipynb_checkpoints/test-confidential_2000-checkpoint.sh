#!/bin/bash
#SBATCH -J confidential_2000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 8-00:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G

# your code goes below
#module load any/python/3.9.9
module load any/python/3.8.3-conda
#conda env create -f environment.yml
conda activate lstm-caise23


srun python ./dg_predictiction.py -a pred_sfx -c confidential_2000 -b "confidential_2000.h5" -v "d_action" -r 1 
srun python ./dg_predictiction.py -a pred_sfx -c confidential_2000 -b "confidential_2000.h5" -v "random_choice" -r 1
srun python ./dg_predictiction.py -a pred_sfx -c confidential_2000 -b "confidential_2000.h5" -v "arg_max" -r 1
#srun python ./dg_predictiction.py -a pred_sfx -c 2012 -b "bpic2012.h5" -v "d_action" -r 1
#srun python ./dg_predictiction.py -a pred_sfx -c 2012 -b "bpic2012.h5" -v "random_choice" -r 1
#srun python ./dg_predictiction.py -a pred_sfx -c 2012 -b "bpic2012.h5" -v "arg_max" -r 1
#srun python ./dg_predictiction.py -a pred_sfx -c 2017 -b "bpic2017.h5" -v "d_action" -r 1
#srun python ./dg_predictiction.py -a pred_sfx -c 2017 -b "bpic2017.h5" -v "random_choice" -r 1
#srun python ./dg_predictiction.py -a pred_sfx -c 2017 -b "bpic2017.h5" -v "arg_max" -r 1
#srun python ./dg_training.py -f bpic2012W.csv -m lstm -e 1 -o bayesian 
