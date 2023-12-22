Niter=10000
Dir=../results/optimize

optFile=newSingleConductance
saveDir=../results/optimize/
for i in `seq $Niter`; do
   python optSynWeight.py -o $optFile --saveDir $saveDir
#    python results/utils/optSynWeight.py --saveDir $Dir &
done
python optSynWeight.py -o $optFile --saveDir $saveDir --saveBest
