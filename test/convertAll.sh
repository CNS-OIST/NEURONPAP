for file in `ls | grep GluTVD | grep .psf`;do
fName=`echo $file | awk -F '.psf' '{print $1}'`
convert "$fName".psf "$fName".pdf
done
