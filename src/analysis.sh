#########################################################
# Zsh Script to run  all analyses in the paper          #
#                                                       #
# It should generate all figures utilized in the paper  #
# as well as other additional plots.                    #
# all results will be zipped in                         #
# FullResults.zip                                       #
#                                      by RJ Nakatani   #
#########################################################

fn_exists() {
  if [[ $(type -w $1 | awk '{print $2}') != "none" ]]; then
    return true
  else
    return false
  fi

}
if fn_exists nproc; then
  np=$(nproc)
  echo "$np processes found; Use half"
  if read -q "choice?Press Y/y to continue."; then
    np=$(expr $np / 2) # Use only half of all processes
  else
    echo "Using All $np"
  fi
else
  np=4 # generic guess for processes
fi

echo "Using parallization num of process:$np"

output=$(date +'%m-%d-%H-%M')out.log
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
if [ ! -d "../results" ]; then
  echo "Directory ../results does not exist"
  mkdir ../results
fi

# rm -r ../results/paperRes
# mkdir ../results/paperRes

# of paps - 1
total=0

for i in $( # for ten random PAPs
  seq 0 $total
); do
  echo "Running K comparison experiments" >>$output
  mpiexec -n $np python NEURONPAP.py --kComp --GluT 0 --stimCount 10 $i  #Fig 1f
  mpiexec -n $np python NEURONPAP.py --kComp --stimGlu --stimCount 10 $i # Fig 2c

  for j in 0.5 10; do # for extracellular potassium condition 0.5 and 10
    echo "seed $i-Ko$j" >>$output
    mpiexec -n $np python NEURONPAP.py -c --stimGlu --GluT 1 --NMDAR 0 --ko $j $i  # Fig 4a
    mpiexec -n $np python NEURONPAP.py -c --stimGlu --GluT 0 --NMDAR 1 --ko $j $i  # Fig 4a
    mpiexec -n $np python NEURONPAP.py -c --stimGaba --GABAR 1 --GluT 0 --ko $j $i # Fig 4a
    echo "Running multi Stim" >>$output
    mpiexec -n $np python NEURONPAP.py -c --stimGlu --GluT 1 --NMDAR 0 --stimCount 10 --ko $j $i  # Fig 4a
    mpiexec -n $np python NEURONPAP.py -c --stimGlu --GluT 0 --NMDAR 1 --stimCount 10 --ko $j $i  # Fig 4a
    mpiexec -n $np python NEURONPAP.py -c --stimGaba --GABAR 1 --GluT 0 --stimCount 10 --ko $j $i # Fig 1ghi 3ab 4abcd
    {
      if (($i == 1)); then
        echo "KO spillover Comparison"
        mpiexec -n $np python NEURONPAP.py --stimK --stimGlu --gluSpill --koComp --stimCount 10 --ko $j $i # Fig 5abcd
        echo "K spillover Comparison"
        mpiexec -n $np python NEURONPAP.py --stimK --gluSpill --stimCount 10 --ko $j $i # Fig 5abcd
        echo "Glu spillover Comparison"
        mpiexec -n $np python NEURONPAP.py --stimGlu --gluSpill --stimCount 10 --ko $j $i # Fig 5abcd
        echo "eK Clamp"
        python NEURONPAP.py --ekComp $i # Fig 2d
      fi
      echo "Making videos and branch attenuation"
      python NEURONPAP.py -v --stimCount 10 --ko $j $i            # Fig 1cd 2a
      python NEURONPAP.py -b --stimCount 10 --ko $j $i            # Fig 1e
      python NEURONPAP.py -b --stimCount 10 --stimGlu --ko $j $i  # Fig 2b
      python NEURONPAP.py -b --stimCount 10 --stimGaba --ko $j $i # Fig 2b
    } >>$output
    # echo "Running KO experiments" >> $output
    # mpiexec -n $np python NEURONPAP.py -c --ko 10 --NMDAR 0 --GluT 0 --stimGlu $i
    # mpiexec -n $np python NEURONPAP.py -c --ko 10 --NMDAR 0 --GluT 0 --stimCount 10 --stimGlu $i
    # mpiexec -n $np python NEURONPAP.py -c --ko 10 --NMDAR 0 --GluT 1 --stimCount 10 --stimGlu $i
  done
done

seed=1

# change value 10 for more or less simultaneous activation
for i in 10; do
  mpiexec -n $np python NEURONPAP.py -c --PAPCount $i --stimCount 10 $seed
done
python NEURONPAP.py --somaVC $seed # Fig 1b
{
  if (($np >= 20)); then
    echo "Phase Plot for default"
    mpiexec -n 10 python NEURONPAP.py --phase $seed & # Fig 4e
    mpiexec -n 10 python NEURONPAP.py --phase --spillOver $seed
    wait
    echo "Phase Plot for multi stim"
    mpiexec -n 10 python NEURONPAP.py --phase --stimCount 10 $seed &
    echo "Phase Plot for KO"
    mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 1 --GluT 0 --GABAR 0 $seed # Fig 4e
    wait
    mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 0 --GluT 1 --GABAR 0 --spillOver $seed & # Fig 4e
    mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 0 --GluT 0 --GABAR 1 $seed               # Fig 4e
    wait
    mpiexec -n 10 python NEURONPAP.py --phase --NMDAR 0 --GluT 1 --GABAR 1 $seed &
    echo "bath experiments"
    mpiexec -n 3 python NEURONPAP.py --bathExperiment $seed
    wait
  else
    echo "Skipped Phase plane analysees and bath experiments"
    echo "They take quite long so run them individually or with more processes"
  fi

} >>$output
# mpiexec -n 2 python experiments.py
# mpiexec -n 1 python experiments.py
zip -rq FullResults.zip ../results/paperRes ../morphResults/video/*.gif ../morphResults/*.psf
