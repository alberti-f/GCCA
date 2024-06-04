#!/bin/bash 
 
# This script sets up a task array with a step size of one. 
 
#SBATCH -J gcca
#SBATCH -p short
#SBATCH --array 1-321%50 
#SBATCH --requeue  
#SBATCH --cpus-per-task=1
#SBATCH -o logs/canproj-%j.out

printf "\n\n\n\n"
echo `date`: Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
echo SLURM_ARRAY_TASK_MIN=${SLURM_ARRAY_TASK_MIN}, SLURM_ARRAY_TASK_MAX=${SLURM_ARRAY_TASK_MAX}, SLURM_ARRAY_TASK_STEP=${SLURM_ARRAY_TASK_STEP} 

module load Python/3.9.5-GCCcore-10.3.0
source /well/margulies/users/bez157/python/gcca-${MODULE_CPU_TYPE}/bin/activate

module load ConnectomeWorkbench/1.5.0-GCCcore-10.3.0


##########################################################################################
DATA=/well/margulies/users/bez157
HCP=/well/win-hcp/HCP-YA/subjectsAll
IDs=$DATA/GCCA/subj_IDs_428.txt
OUT=$DATA/GCCA/Output_428

python Smooth_and_concat.py ${SLURM_ARRAY_TASK_ID} 6 $IDs $HCP $OUT
python Canonical_projections.py ${SLURM_ARRAY_TASK_ID} 3 $IDs $HCP $OUT

echo `date`: task complete 
