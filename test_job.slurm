#!/bin/bash --login

#SBATCH --job-name="ConstructAllW5321 Test"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --account=ta135-shudson
#SBATCH --partition=standard
#SBATCH --qos=lowpriority

# Load the Python module
module load cray-python

python "ConstructAllW5321 - Copy"
