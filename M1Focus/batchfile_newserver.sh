#!/bin/sh
#SBATCH -J M1_sim 
#SBATCH -o  ./stdout/M1_sim.o%j.out
#SBATCH -e  ./stdout/M1_sim.e%j.error
#SBATCH -t 0-48:00:00  # days-hours:minutes

#SBATCH -N 1
#SBATCH -n 50 # used for MPI codes, otherwise leave at '1'
##SBATCH --ntasks-per-node=1  # don't trust SLURM to divide the cores evenly
##SBATCH --cpus-per-task=1  # cores per task; set to one if using MPI
##SBATCH --exclusive  # using MPI with 90+% of the cores you should go exclusive
#SBATCH --mem-per-cpu=4G  # memory per core; default is 1GB/core

## send mail to this address, alert at start, end and abortion of execution
##SBATCH --mail-type=ALL
##SBATCH --mail-user=zc63@mail.missouri.edu

START=$(date)

unset DISPLAY
## mpirun nrniv -mpi MC_main_small_forBeta_shortburstensamble.hoc #srun
mpirun nrniv -mpi -python run_network.py config.json #srun

END=$(date)

echo "Started running at $START."
echo "Done running simulation at $END"