#!/bin/bash
#SBATCH -J adni_bs16_patch7-2_ep1000
#SBATCH -N 1
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --mem=128gb
#SBATCH --ntasks=1
#SBATCH --mail-user=hui.zheng@tum.de
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00
#SBATCH -o %x.%j.%N.out

source ~/.bashrc  # activate miniconda
source ~/miniconda3/bin/activate mae # activate your environment

cd ~/ijepa/

export WANDB_API_KEY=9b379393a7a65969e05ab4e01683be3b8770aabf

srun python main.py \
  --fname configs/adni_vitb16.yaml \
  --devices cuda:0 