#!/bin/bash
#SBATCH --job-name=mcmc_chains
#SBATCH --array=0-7             
#SBATCH --output=mcmc_chains_%A_%a.out
#SBATCH --error=mcmc_chains_%A_%a.err
#SBATCH --time=4-16:00:00          
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=6          
#SBATCH --mem=12G                    
#SBATCH --partition=regular          

# Load Python 3.9 module
# module load python/3.9.19

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6

# Run MCMC script
python3 -u run_simulation.py