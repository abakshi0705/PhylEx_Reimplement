#!/bin/bash
#SBATCH --job-name=arviz_analysis
#SBATCH --output=arviz_%j.out
#SBATCH --error=arviz_%j.err
#SBATCH --time=0-01:00:00         
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=regular

# Run the analysis script
python analyze_convergence.py