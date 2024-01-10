#!/bin/bash

# USAGE
# ./sbatch transcribe.sh <bucket-name> <path/to/recording> <path/to/output>

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling

#SBATCH --job-name=transcribe                                   # sets the job name
#SBATCH --output=job_logs/transcribe.out.%j                              # indicates a file to redirect STDOUT to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --error=job_logs/transcribe.out.%j                               # indicates a file to redirect STDERR to; %j is the jobid. If set, must be set to a file instead of a directory or else submission will fail.
#SBATCH --time=00:30:00                                         # how long you would like your job to run; format=hh:mm:ss
#SBATCH --qos=default                                           # set QOS, this will determine what resources can be requested
#SBATCH --mem=8gb                                               # memory required by job; if unit is not specified MB will be assumed
#SBATCH --gres=gpu:1

# load necessarily modules
module load Python3/3.10.10
module load ffmpeg/6.0
module load cuda

# ensure using right environment
source ~/whisperx/bin/activate

# copy specified file from umobj to local
cpobj $1:$2 ~/input.mp3

# submit transcription job
srun python transcribe.py ~/input.mp3 ~/transcription.txt

# copy transcription text to umobj
cpobj ~/transcription.txt $1:$3

# cleanup
rm ~/input.mp3
rm ~/transcription.txt