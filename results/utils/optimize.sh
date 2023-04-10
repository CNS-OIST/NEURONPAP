Niter=2000
Dir=./results/optimize

optFile=original_refit_A2A3A1DELTA_RelativeShape_SmallB2B3_smalltau
saveDir=./results/optimize/
for i in `seq $Niter`; do
   python results/utils/optSynWeight.py -o $optFile --saveDir $saveDir
#    python results/utils/optSynWeight.py --saveDir $Dir &
done
python results/utils/optSynWeight.py -o $optFile --saveDir $saveDir --saveBest
