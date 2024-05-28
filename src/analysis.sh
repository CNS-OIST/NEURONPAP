#########################################################
# Zsh Script to run  all analyses in the paper          #
#                                                       #
# It should generate all figures utilized in the paper  #
# as well as other additional plots.                    #
# all results will be zipped in                         #
# FullResults.zip                                       #
#                                      by RJ Nakatani   #
#########################################################

np=`nproc`
output=`date +'%m-%d-%H-%M'`out.log
outputDir="../outlog/"
output=$outputDir$output
touch $output

nrnivmodl neuronMOD
rm ../morphResults/video/* intermediaryData/*
rm -r ../results/paperRes
mkdir ../results/paperRes

for i in `seq 0 9`;
    for j in 0.5 8; do # 0,0.5,8
        echo "seed $i-Ko$j" >> $output
        mpiexec -n $np python NEURONPAP.py -c --ko $j $i
        echo "Running multi Stim" >> $output
        mpiexec -n $np python NEURONPAP.py -c --stimCount 10 --ko $j $i
        {
            if (( $i == 1 )); then
                     echo "KO spillover Comparison"
                     mpiexec -n 20 python NEURONPAP.py --gluSpill --koComp --stimCount 10 --ko $j $i
                     echo "eK Clamp"
                     python NEURONPAP.py --ekComp $i
            fi
            echo "Making videos and branch attenuation"
            python NEURONPAP.py -v --stimCount 10 --ko $j $i
            python NEURONPAP.py -b --stimCount 10 --ko 10 $i
            python NEURONPAP.py -b --stimCount 10 --stimGlu --ko $j $i
        } >> $output
        echo "Running KO experiments" >> $output
        mpiexec -n $np python NEURONPAP.py -c --ko 10 --NMDAR 0 --GluT 0 $i
        mpiexec -n $np python NEURONPAP.py -c --ko 10 --NMDAR 0 --GluT 0 --stimCount 10 $i
    done
    echo "Running K comparison experiments" >> $output
    mpiexec -n $np python NEURONPAP.py --kComp --stimCount 10 $i
    mpiexec -n $np python NEURONPAP.py --kComp --stimGlu --stimCount 10 $i
done

seed=1
# for i in `seq 0 5`; do
#     mpiexec -n $np python NEURONPAP.py -c --stim --stimK --stimGlu --stimCount 10 --delay $i $seed
# done

# change value 10 for more simultaneous activation
for i in 10; do
    mpiexec -n $np python NEURONPAP.py -c --PAPCount $i --stimCount 10 $seed
done
python NEURONPAP.py --vClamp $seed
{
    echo "Phase Plot for default"
    mpiexec -n 10 python NEURONPAP.py --phase $seed
    mpiexec -n 10 python NEURONPAP.py --phase --spillOver $seed
    echo "Phase Plot for multi stim"
    mpiexec -n 10 python NEURONPAP.py --phase --stimCount 10 $seed
    echo "Phase Plot for KO"
    mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 1 --GluT 0 $seed
    mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 0 --GluT 1 $seed
    mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 0 --GluT 0 $seed
    mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 0 --GluT 0 --spillOver $seed
} >> $output
zip -r FullResults.zip ../results/paperRes ../morphResults/video/*.gif ../morphResults/*.psf
