np=`nproc`
output=`date +'%m-%d-%H-%M'`out.log
outputDir="../outlog/"
output=$outputDir$output
touch $output

nrnivmodl neuronMOD
rm ../morphResults/video/* intermediaryData/*
rm -r ../results/paperRes
mkdir ../results/paperRes

for i in `seq 0 4`; do # seq 0 9
    for j in 0.5 1 2 4 8; do # 0,0.5,8
        echo "seed $i-Ko$j" >> $output
        mpiexec -n $np python NEURONPAP.py -c --ko $j $i
        # mpiexec -n $np python NEURONPAP.py -c --ko $j --stimCount 2 $i
        echo "Running multi Stim" >> $output
        mpiexec -n $np python NEURONPAP.py -c --ko $j $i
        # mpiexec -n $np python NEURONPAP.py -c --stimCount 5 --ko $j $i
        mpiexec -n $np python NEURONPAP.py -c --stimCount 10 --ko $j $i
        {
            if (( $i == 0 )); then
                     echo "KO Comparison"
                     mpiexec -n 10 python NEURONPAP.py --koComp --ko $j $i
            fi
            echo "Making videos and branch attenuation"
            python NEURONPAP.py -v --ko $j $i
            python NEURONPAP.py -b --stimCount 10 --ko $j $i
        } >> $output
        echo "Running KO experiments" >> $output
        mpiexec -n $np python NEURONPAP.py -c --ko $j --NMDAR 0 --GluT 1 $i
        mpiexec -n $np python NEURONPAP.py -c --ko $j --NMDAR 1 --GluT 0 $i
        mpiexec -n $np python NEURONPAP.py -c --ko $j --NMDAR 0 --GluT 0 $i
    done
    mpiexec -n $np python NEURONPAP.py --kComp $i
    # mpiexec -n $np python NEURONPAP.py --kComp --stimCount 5 $i
    mpiexec -n $np python NEURONPAP.py --kComp --stimCount 10 $i
    python NEURONPAP.py --ekComp $i
done

seed=1
# echo "Running No Glutamate stimulus" >> $output
# mpiexec -n $np python NEURONPAP.py -c --stim --stimK $seed
# mpiexec -n $np python NEURONPAP.py -c --stim --stimGlu $seed

# for i in `seq 0 5`; do
#     mpiexec -n $np python NEURONPAP.py -c --stim --stimK --stimGlu --delay $i $seed
# done

# for i in `seq 2 10`; do
#     mpiexec -n $np python NEURONPAP.py -c --PAPCount $i $seed
# done
python NEURONPAP.py --vClamp $seed
{
    echo "Phase Plot for default"
    mpiexec -n 10 python NEURONPAP.py --phase $seed &
    echo "Phase Plot for two stim"
    # mpiexec -n 10 python NEURONPAP.py --phase --stimCount 5 $seed
    mpiexec -n 10 python NEURONPAP.py --phase --stimCount 10 $seed
    echo "Phase Plot for KO"
    mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 1 --GluT 0 $seed
    mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 0 --GluT 1 $seed
    mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 0 --GluT 0 $seed
} >> $output
zip -r FullResults.zip ../results/paperRes video/*.gif 
