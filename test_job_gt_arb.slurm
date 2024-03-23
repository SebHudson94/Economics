#!/bin/bash --login

#SBATCH --job-name="GTArb"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --account=ta135-shudson
#SBATCH --partition=standard
#SBATCH --qos=short

source /work/ta135/ta135/shudson/myvenv/bin/activate

# Load the Python module
module load cray-python

export PYTHONUSERBASE=/work/ta135/ta135/shudson/.local
export PATH=$PYTHONUSERBASE/bin:$PATH

python GT-Arbitrage-Fast.py
