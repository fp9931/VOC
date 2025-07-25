#!/bin/bash
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1          # 1 tasks per node
#SBATCH --time=24:00:00                 # time limits: 24 hours
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --error=Bulbar_completo_rfe.err       # standard error file
#SBATCH --output=Bulbar_completo_rfe.out      # standard output file
#SBATCH --account=IscrC_AIM-ORAL     # account name

python Bulbar_completo_rfe.py