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

echo "Do you wish to create new videos (may take time)?"
select yn in "Yes" "No"; do
  case $yn in
  Yes)
    video_bool=1
    echo "Selected $yn"
    break
    ;;
  No)
    video_bool=0
    echo "Selected $yn"
    break
    ;;
  esac
done

echo "Do you wish to run in vivo traces (may take time)?"
select yn in "Yes" "No"; do
  case $yn in
  Yes)
    invivo_bool=1
    echo "Selected $yn"
    break
    ;;
  No)
    invivo_bool=0
    echo "Selected $yn"
    break
    ;;
  esac
done

# of paps
total=1
seed=1
# insert in order of figures.
#
# Panel A: model cartoon
mpiexec -n 11 python NEURONPAP.py --somaClamp $seed # Fig 1bc
if (($invivo_bool == 1)); then
mpiexec -n 6 python experiments.py                  # Fig 1 de, Fig 2,A,B 
fi
python experiments.py                                
#mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py --freqComp --stimCount 10 --NMDAR 1 --GluT 1 --GABAR 0 $seed
mpiexec -n 3 python experiments.py # fig 3,4,5 experiment fit
for i in $( # for ten random PAPs
  seq 1 $total
); do
  echo "Running K comparison experiments" >>$output
  mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py --kComp --GluT 0 --stimCount 10 $i #Fig 2CDE
  mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py --kComp --GluT 1 --stimCount 10 $i #Fig 4BCD
  mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py --kComp --GluT 1 --stimCount 10 --intraDiff $i #Fig 4BCD
  for j in 0.5 22; do                                                                    # for extracellular potassium condition 0.5 and 10
    echo "seed $i-Ko$j" >>$output
    for k in 1 10; do # for fig 5stimCoutn
      if (($j == 0.5)); then 
        mpiexec -n 10 --use-hwthread-cpus python NEURONPAP.py --shellExp --GABAR 0 --NMDAR 1 --Glu 0 --stimCount $k --stimGlu 1
        mpiexec -n 10 --use-hwthread-cpus python NEURONPAP.py --shellExp --GABAR 0 --NMDAR 0 --Glu 1 --stimCount $k --stimGlu 1
        mpiexec -n 10 --use-hwthread-cpus python NEURONPAP.py --shellExp --GABAR 1 --NMDAR 0 --Glu 0 --stimCount $k --stimGaba 1
        mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py -c --stimGaba --GABAR 1 --GluT 0 --stimCount $k --ko $j $i                     
        mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py -c --stimGlu --GluT 1 --NMDAR 1 --stimCount $k --ko $j $i                     
        python NEURONPAP.py -b --stimCount $k --stimGlu --ko $j $i                                                             # 
        python NEURONPAP.py -b --stimCount $k --stimGaba --ko $j $i                                                            #
       elif (($j == 10)); then
        mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py -c --stimGlu --GluT 1 --NMDAR 0 --stimCount $k --ko $j $i                         
        mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py -c --stimGlu --GluT 0 --NMDAR 0 --NKA 1 --stimCount $k --ko $j $i                
     fi
      #mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py -c --GluT 0 --NMDAR 0 --GABAR 0 --GAP 1 --stimCount $k --ko $j $i            
      #mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py -c --stimGlu --PAPCount 10 --GluT 1 --NMDAR 1 --GABAR 0 --stimCount $k --ko $j $i
      if (($k == 10)); then
  # fig 3 ab
        python NEURONPAP.py -s --NMDAR 0 --GABAR 0 --GluT 0 --stimCount $k --ko $j $i
        python NEURONPAP.py -s --NMDAR 0 --GABAR 0 --GluT 0 --stimCount $k --ko $j --intraDiff $i
        if (($video_bool == 1)); then
          python NEURONPAP.py -v --stimCount $k --ko $j $i            #
          python NEURONPAP.py -v --stimCount $k --stimGlu --ko $j $i  #
          python NEURONPAP.py -v --stimCount $k --stimGaba --ko $j $i # Fig 4A
        fi
        #python NEURONPAP.py -b --stimCount $k --ko $j $i                                                                       #
  # fig 56
       #mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py --stimK --stimGlu --gluSpill --koComp --stimCount $k --ko $j $i #
        #mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py --stimGlu --gluSpill --stimCount $k --ko $j $i                  # 
      fi
    done
  done
  #echo "Running KO experiments" >>$output
  # fig 56
  #mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py -c --ko 10 --NMDAR 0 --GluT 0 --stimGlu --stimK $i
  #mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py -c --ko 10 --NMDAR 0 --GluT 0 --stimCount 10 --stimGlu --stimK $i
  #mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py -c --ko 10 --NMDAR 0 --GluT 1 --stimCount 10 --stimGlu --stimK $i
done

{
  if (($np >= 6)); then
    if (($np >= 12)); then
      np_phase=12
    else
      np_phase=6
    fi

  # fig 56
    #mpiexec -n $np_phase python NEURONPAP.py --phase --NMDAR 0 --GluT 0 --GABAR 0 $seed
    #mpiexec -n $np_phase python NEURONPAP.py --phase --NMDAR 0 --GluT 1 --GABAR 0 $seed # 
    mpiexec -n $np_phase python NEURONPAP.py --phase --NMDAR 1 --GluT 1 --GABAR 0 $seed
    mpiexec -n $np_phase python NEURONPAP.py --phase --NMDAR 0 --GluT 0 --GABAR 1 $seed
    #mpiexec -n $np_phase python NEURONPAP.py --phase --NMDAR 0 --GluT 0 --GABAR 0 --NKA 1 $seed
  else
    echo "Skipped Phase plane analysees and bath experiments"
    echo "They take quite long so run them individually or with more processes"
  fi

} >>$output
# fig 8
#mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py --testPhys 1
mpiexec -n $np --use-hwthread-cpus python NEURONPAP.py --testPhys --intraDiff 1
mpiexec -n 2 python experiments.py
mpiexec -n 1 python experiments.py
mpiexec -n $np python experiments.py
mpiexec -n $np python experiments.py
zip -rq FullResults.zip ../results/paperRes ../morphResults/video/*.gif ../morphResults/*.psf

# if uv exists use uv environment
if command -v uv &>/dev/null; then
  echo "Found UV deactivating uv environment"
  deactivate
fi
