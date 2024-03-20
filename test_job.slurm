#!/bin/bash --login

#SBATCH --name=ConstructAllW5321 Test
#SBATCH --node=1
#SBATCH --ntasks-per-node=128
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time==00:10:00
#SBATCH --account=ta135
#SBATCH --partition=standard
#SBATCH --qos=lowpriority

# Setup the batch environment
module load epcc-job-env

# Load the Python module
module load cray-python

python ConstructAllW5321 - Copy
