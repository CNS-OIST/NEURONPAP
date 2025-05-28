#!/bin/bash
#########################################################
# SLURM Script to run  all analyses in the paper          #
#                                                       #
# It should generate all figures utilized in the paper  #
# as well as other additional plots.                    #
# all results will be zipped in                         #
# FullResults.zip                                       #
#                                      by RJ Nakatani   #
#########################################################

## give a name to your job
#SBATCH --job-name=NEURONPAP

## number of cores for your simulation,
## for parallel job array about 20
#SBATCH --ntasks=80
#SBATCH --nodes=1


## Save output file as seed and endt name
#SBATCH --output=../outlog/NEURONPAP_%j.out

## you can replace compute with short if your simulation time is below 1 day
#SBATCH --partition=compute

## how much memory per core
#SBATCH --mem-per-cpu=3g

## maximum time for your simulation, in DAY-HOUR:MINUTE:SECOND
#SBATCH --time=0-10:00:0



echo "Using parallization num of process:"$SLURM_NTASKS



module use /apps/unit/DeSchutterU/.modulefiles
module load neuron/2025_03_19_8.2.6

nrnivmodl neuronMOD

# Prepare directories
if [ ! -d "../results" ];then
    echo "Directory ../results does not exist"
    mkdir ../results
fi

# rm -r ../results/paperRes
# mkdir ../results/paperRes

# of paps - 1
total=0

for i in `seq 0 $total`; do # for ten random PAPs
    echo "Running K comparison experiments" 
    srun --mpi=pmix  python NEURONPAP.py --kComp --stimCount 10 $i #Fig 1f
    srun --mpi=pmix  python NEURONPAP.py --kComp --stimGlu --stimCount 10 $i # Fig 2c
    
    for j in 0.5 10; do # for extracellular potassium condition 0.5 and 10
        echo "seed $i-Ko$j" 
        srun --mpi=pmix  python NEURONPAP.py -c --stimGlu --GluT 1 --NMDAR 0 --ko $j $i # Fig 4a
        srun --mpi=pmix  python NEURONPAP.py -c --stimGlu --GluT 0 --NMDAR 1 --ko $j $i # Fig 4a
        srun --mpi=pmix  python NEURONPAP.py -c --stimGaba --ko $j $i # Fig 4a
        echo "Running multi Stim" 
        srun --mpi=pmix  python NEURONPAP.py -c --stimGlu --GluT 1 --NMDAR 0 --stimCount 10 --ko $j $i # Fig 4a
        srun --mpi=pmix  python NEURONPAP.py -c --stimGlu --GluT 0 --NMDAR 1 --stimCount 10 --ko $j $i # Fig 4a
        srun --mpi=pmix  python NEURONPAP.py -c --stimGaba --stimCount 10 --ko $j $i # Fig 1ghi 3ab 4abcd
        {
            if (( $i == 1 )); then
                     echo "KO spillover Comparison"
                     srun --mpi=pmix  python NEURONPAP.py --stimK --stimGlu --gluSpill --koComp --stimCount 10 --ko $j $i # Fig 5abcd
                     echo "K spillover Comparison"
                     srun --mpi=pmix  python NEURONPAP.py --stimK --gluSpill --stimCount 10 --ko $j $i # Fig 5abcd
                     echo "Glu spillover Comparison"
                     srun --mpi=pmix  python NEURONPAP.py --stimGlu --gluSpill --stimCount 10 --ko $j $i # Fig 5abcd
                     echo "eK Clamp"
                     python NEURONPAP.py --ekComp $i # Fig 2d
            fi
            echo "Making videos and branch attenuation"
            python NEURONPAP.py -v --stimCount 10 --ko $j $i # Fig 1cd 2a
            python NEURONPAP.py -b --stimCount 10 --ko $j $i # Fig 1e
            python NEURONPAP.py -b --stimCount 10 --stimGlu --ko $j $i  # Fig 2b
            python NEURONPAP.py -b --stimCount 10 --stimGaba --ko $j $i  # Fig 2b
        } 
        # echo "Running KO experiments" 
        # srun --mpi=pmix  python NEURONPAP.py -c --ko 10 --NMDAR 0 --GluT 0 --stimGlu $i 
        # srun --mpi=pmix  python NEURONPAP.py -c --ko 10 --NMDAR 0 --GluT 0 --stimCount 10 --stimGlu $i
        # srun --mpi=pmix  python NEURONPAP.py -c --ko 10 --NMDAR 0 --GluT 1 --stimCount 10 --stimGlu $i
    done
done

seed=1

# change value 10 for more or less simultaneous activation
for i in 10; do
    srun --mpi=pmix python NEURONPAP.py -c --PAPCount $i --stimCount 10 $seed
done
python NEURONPAP.py --vClamp $seed # Fig 1b
{
       echo "Phase Plot for default"
       mpiexec -n 10 python NEURONPAP.py --phase $seed & # Fig 4e
       mpiexec -n 10 python NEURONPAP.py --phase --spillOver $seed &
       echo "Phase Plot for multi stim"
       mpiexec -n 10 python NEURONPAP.py --phase --stimCount 10 $seed &
       echo "Phase Plot for KO"
       mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 1 --GluT 0 --GABAR 0 $seed & # Fig 4e
       mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 0 --GluT 1 --GABAR 0 --spillOver $seed & # Fig 4e
       mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 0 --GluT 0 --GABAR 1 $seed &# Fig 4e
       mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 0 --GluT 1 --GABAR 1 $seed &
       echo "bath experiments"
       mpiexec -n 3 python NEURONPAP.py --bathExperiment $seed
       wait

} 
# mpiexec -n 2 python experiments.py
# mpiexec -n 1 python experiments.py
