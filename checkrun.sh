#!/bin/bash

#SBATCH --job-name=postrun
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1
#SBATCH --constraint=intel
#SBATCH --mail-user=rishabh.dutta@kaust.edu.sa
#SBATCH --mail-type=ALL

# Print some job information
echo
echo "My execution hostname is: $(hostname -s)."
echo "I am job $SLURM_JOB_ID, a member of the job array $SLURM_ARRAY_JOB_ID"
echo "and my task ID is $SLURM_ARRAY_TASK_ID"
echo

cd /scratch/dragon/intel/duttar/python_gorkhaJul19

#sh close_processes.sh
#export arrayindex=$SLURM_ARRAY_TASK_ID

module purge all 
module load python/3.6.2

#sleep $[( $RANDOM % 30 )]s
python stage_1_run.py
