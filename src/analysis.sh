#########################################################
# Zsh Script to run  all analyses in the paper          #
#                                                       #
# It should generate all figures utilized in the paper  #
# as well as other additional plots.                    #
# all results will be zipped in                         #
# FullResults.zip                                       #
#                                      by RJ Nakatani   #
#########################################################

fn_exists () {
    if [[ `type -w $1 | awk '{print $2}'` != "none" ]]; then
        return true
    else
        return false
    fi

}
if fn_exists nproc;then
    np=`nproc`
    np=`expr $np / 2` # Use only half of all processes
else
    np=4 # generic guess for processes
fi

echo "Using parallization num of process:$np"


output=`date +'%m-%d-%H-%M'`out.log
outputDir="../outlog/"
output=$outputDir$output
touch $output

# requires nrnivmodl installation
if fn_exists nrnivmodl; then
    nrnivmodl neuronMOD
else
    echo "No NEURON; INSTALL NEURON"
fi

# Prepare directories
rm ../morphResults/video/* intermediaryData/*
if [ ! -d "../results" ];then
    echo "Directory ../results does not exist"
    mkdir ../results
fi

rm -r ../results/paperRes
mkdir ../results/paperRes

for i in `seq 0 9`; do # for ten random PAPs
    for j in 0.5 10; do # for extracellular potassium condition 0.5 and 10
        echo "seed $i-Ko$j" >> $output
        mpiexec -n $np python NEURONPAP.py -c --ko $j $i # Fig 4a
        echo "Running multi Stim" >> $output
        mpiexec -n $np python NEURONPAP.py -c --stimCount 10 --ko $j $i # Fig 1ghi 3ab 4abcd
        {
            if (( $i == 1 )); then
                     echo "KO spillover Comparison"
                     mpiexec -n $np python NEURONPAP.py --gluSpill --koComp --stimCount 10 --ko $j $i # Fig 5abcd
                     echo "eK Clamp"
                     python NEURONPAP.py --ekComp $i # Fig 2d
            fi
            echo "Making videos and branch attenuation"
            python NEURONPAP.py -v --stimCount 10 --ko $j $i # Fig 1cd 2a
            python NEURONPAP.py -b --stimCount 10 --ko $j $i # Fig 1e
            python NEURONPAP.py -b --stimCount 10 --stimGlu --ko $j $i  # Fig 2b
        } >> $output
        echo "Running KO experiments" >> $output
        mpiexec -n $np python NEURONPAP.py -c --ko 10 --NMDAR 0 --GluT 0 $i 
        mpiexec -n $np python NEURONPAP.py -c --ko 10 --NMDAR 0 --GluT 0 --stimCount 10 $i
        mpiexec -n $np python NEURONPAP.py -c --ko 10 --NMDAR 0 --GluT 1 --stimCount 10 $i
    done
    echo "Running K comparison experiments" >> $output
    mpiexec -n $np python NEURONPAP.py --kComp --stimCount 10 $i #Fig 1f
    mpiexec -n $np python NEURONPAP.py --kComp --stimGlu --stimCount 10 $i # Fig 2c
done

seed=1

# change value 10 for more or less simultaneous activation
for i in 10; do
    mpiexec -n $np python NEURONPAP.py -c --PAPCount $i --stimCount 10 $seed
done
python NEURONPAP.py --vClamp $seed # Fig 1b
{
    echo "Phase Plot for default"
    mpiexec -n 10 python NEURONPAP.py --phase $seed # Fig 4e
    mpiexec -n 10 python NEURONPAP.py --phase --spillOver $seed
    echo "Phase Plot for multi stim"
    mpiexec -n 10 python NEURONPAP.py --phase --stimCount 10 $seed
    echo "Phase Plot for KO"
    mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 1 --GluT 0 $seed # Fig 4e
    mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 0 --GluT 1 $seed # Fig 4e
    mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 0 --GluT 0 $seed # Fig 4e
    mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 0 --GluT 0 --spillOver $seed
} >> $output
zip -r FullResults.zip ../results/paperRes ../morphResults/video/*.gif ../morphResults/*.psf
