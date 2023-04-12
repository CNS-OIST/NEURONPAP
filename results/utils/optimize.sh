Niter=10000
Dir=./results/optimize

optFile=original_refit_Relative
saveDir=./results/optimize/
for i in `seq $Niter`; do
   python results/utils/optSynWeight.py -o $optFile --saveDir $saveDir
#    python results/utils/optSynWeight.py --saveDir $Dir &
done
python results/utils/optSynWeight.py -o $optFile --saveDir $saveDir --saveBest
