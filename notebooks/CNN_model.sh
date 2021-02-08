#!/bin/bash                                                                                                                                                                                                       \   #SBATCH --partition=tibet --qos=normal                                                                                                     \
#SBATCH --time=06:00:00                                             \
#SBATCH --nodes=4                                                                                                                                                                                              \
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
# only use the following on partition with GPUs                                                                                                                                                                    #SBATCH --gres=gpu:1
#SBATCH --job-name="sample"
#SBATCH --output=sample-%j.out
#SBATCH --mem-per-cpu=32G
# only use the following if you want email notification                                                                                                                                                            ####SBATCH --mail-user=youremailaddress                                                                                                                                                                            ####SBATCH --mail-type=ALL

# list out some useful information (optional)                                                                                                                                                                      echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)                                                                                                                                                   NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

now=$(date +"%y%m%d-%H%M")
name=$("model-" + SLURM_JOBID) 
logpath="/atlas/u/${USER}/data/scaling-kilns/log"
mkdir -p $logpath
logfile="$logpath/${SLURM_JOBID}.out"

echo "Writing to ${logfile}"
# can try the following to list out which GPU you have access to
python /sailhome/${USER}/scaling-kilns/kiln-scaling/notebooks/CNN_model.py > ${logfile}
# python /sailhome/atlas/u/mliu356/scaling-kilns/kiln-scaling/notebooks/get_tiled_data_from_tiff_hdf5.py                                                                                                          #srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done                                                                                                                                                                                                             echo "Done"
