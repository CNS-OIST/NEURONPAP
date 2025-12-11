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
check_modlunit() {
  for file in $(ls neuronMOD | grep .mod); do
    uv run modlunit neuronMOD/$file
  done
}
if fn_exists nproc; then
  np=$(nproc)
  echo "$np processes found; Use half"
  while true; do
    if read -q "choice?Press Y/y to continue."; then
      np=$(expr $np / 2) # Use only half of all processes
      break
    else
      suggestion=$(nproc)
      echo ""
      reg='^[+]?\d+([.]\d+)?$'
      while [[ ($suggestion -ge $np) && ! ($suggestion =~ $reg) ]]; do
        read "suggestion?How many processes do you want to use?"
      done
      echo "Selected $suggestion processes"
      np=$suggestion
      break
    fi
  done
else
  np=4 # generic guess for processes
fi

echo "Using parallization num of process:$np"

output=$(date +'%m-%d-%H-%M')out.log
outputDir="../outlog/"
output=$outputDir$output
touch $output

# if uv exists use uv environment
if command -v uv &>/dev/null; then
  echo "Found UV using uv environment"
  source .venv/bin/activate
fi

# requires nrnivmodl installation
if fn_exists nrnivmodl; then
  nrnivmodl neuronMOD
  check_modlunit
else
  echo "No NEURON; INSTALL NEURON"
fi

# Prepare directories
echo "remove intermediary files?"
rm -rI ../morphResults/video/ intermediaryData/*
if [ ! -d "../results" ]; then
  echo "Directory ../results does not exist"
  mkdir ../results
fi

echo "remove results?"
rm -rI ../results/paperRes
mkdir ../results/paperRes

# of paps
total=1
seed=1
# insert in order of figures.
#
# figure 1
# Panel A: model cartoon
# Panel C gen in NEURON
# Panel D gen in NEURON
mpiexec -n 10 python NEURONPAP.py --somaVC $seed                         # Fig 1b
mpiexec -n 5 python experiments.py                                       # Fig 1 EFH
python NEURONPAP.py -s --GABAR 0 --NMDAR 0 --GluT 0 --stimCount 10 $seed # Fig1 G
python experiments.py                                                    # Fig 1 I
python NEURONPAP.py -s --NMDAR 0 --GABAR 0 --GluT 0 --stimCount 10 --ko 16 $seed
mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py --freqComp --stimCount 10 --NMDAR 1 --GluT 1 --GABAR 0 $seed

mpiexec -n 3 python experiments.py # fig 3EF 5D
for i in $( # for ten random PAPs
  seq 1 $total
); do
  # fig 2,3
  echo "Running K comparison experiments" >>$output
  mpiexec -n $np python NEURONPAP.py --kComp --GluT 0 --stimCount 10 $i  #Fig 2ABCD
  mpiexec -n $np python NEURONPAP.py --kComp --stimGlu --stimCount 10 $i # Fig 3CD
  for j in 0.5 10; do                                                    # for extracellular potassium condition 0.5 and 10
    echo "seed $i-Ko$j" >>$output
    for k in 1 10; do                                                                                                                # for stimCoutn
      mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py -c --stimGlu --GluT 1 --NMDAR 0 --stimCount $k --ko $j $i               # Fig 3A
      mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py -c --stimGaba --GABAR 1 --GluT 0 --stimCount $k --ko $j $i              # Fig 4CD
      mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py -c --stimGlu --GluT 1 --NMDAR 1 --stimCount $k --ko $j $i               # Fig 5AC
      mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py -c --stimGlu --PAPCount 10 --GluT 1 --NMDAR 1 --stimCount $k --ko $j $i # Fig 5AC
      if (($k == 10)); then
        python NEURONPAP.py -v --stimCount $k --ko $j $i                                                      #
        python NEURONPAP.py -b --stimCount $k --ko $j $i                                                      #
        python NEURONPAP.py -v --stimCount $k --stimGlu --ko $j $i                                            #
        python NEURONPAP.py -b --stimCount $k --stimGlu --ko $j $i                                            # Fig 5B
        python NEURONPAP.py -v --stimCount $k --stimGaba --ko $j $i                                           # Fig 4A
        python NEURONPAP.py -b --stimCount $k --stimGaba --ko $j $i                                           # Fig 4B
        mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py --stimGlu --gluSpill --stimCount $k --ko $j $i # Fig 5abcd
      fi
    done
    if (($i == 1)); then
      echo "KO spillover Comparison"
      mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py --stimK --stimGlu --gluSpill --koComp --stimCount 10 --ko $j $i # Fig 6A
      echo "Glu spillover Comparison"
      mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py --stimGlu --gluSpill --stimCount 10 --ko $j $i # Fig 4B
    fi
    echo "Running KO experiments" >>$output
    mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py -c --ko 10 --NMDAR 0 --GluT 0 --stimGlu $i
    mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py -c --ko 10 --NMDAR 0 --GluT 0 --stimCount 10 --stimGlu $i
    mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py -c --ko 10 --NMDAR 0 --GluT 1 --stimCount 10 --stimGlu $i
  done
done

{
  if (($np >= 6)); then
    mpiexec -n 6 python NEURONPAP.py --phase --NMDAR 0 --GluT 1 --GABAR 0 $seed # Fig 6BCD
    mpiexec -n 6 python NEURONPAP.py --phase --NMDAR 0 --GluT 1 --GABAR 0 --spillOver $seed
    mpiexec -n 6 python NEURONPAP.py --phase --NMDAR 1 --GluT 1 --GABAR 0 $seed
    mpiexec -n 6 python NEURONPAP.py --phase --NMDAR 0 --GluT 0 --GABAR 1 $seed
  else
    echo "Skipped Phase plane analysees and bath experiments"
    echo "They take quite long so run them individually or with more processes"
  fi

} >>$output
mpiexec -n 2 python experiments.py
mpiexec -n 1 python experiments.py
zip -rq FullResults.zip ../results/paperRes ../morphResults/video/*.gif ../morphResults/*.psf

# if uv exists use uv environment
if command -v uv &>/dev/null; then
  echo "Found UV deactivating uv environment"
  deactivate
fi
