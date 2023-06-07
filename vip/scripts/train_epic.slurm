#!/bin/bash
#SBATCH -J VIP-epic
#SBATCH --comment "Traing Value Implicit Pretraining on EPIC"

#SBATCH --qos level0
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --gres=gpu
#SBATCH --partition=gpu
# SBATCH -w lambda-hyperplane01
#SBATCH --account=research
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-10:00:00

#SBATCH -o "/scratch/lipuhao/dev/MAH/CodeRepo/vip/vipoutput/slurms/%x.%j.out"
#SBATCH -e "/scratch/lipuhao/dev/MAH/CodeRepo/vip/vipoutput/slurms/%x.%j.err"

# HOME=/home/lipuhao
name=$1
agentago=$2

python train_vip.py --config-name=config_vip_epickitchen batch_size=64 experiment=$name agentago=$agentago dataset=epic-kitchen datapath=/scratch/generalvision/EgoCentric/EPIC-KITCHEN