#!/bin/bash

juego=prueba_nuevo_calculo_pred_en_dwmcml
DIR=/tmp/$juego

DATASET1=enron
DATASET2=20ng
DATASET3=mediamill

datasets=($DATASET2)
#datasets2=(tmc2007 scene yeast)
#models=(lcht br_nb cc_nb)
models=(dwmc_br)

for d in "${datasets[@]}"
do 
	for m in "${models[@]}" 
	do 
		output=$DIR/"$d"_"$m"_"$juego".log
		python3 classifier.py -d $d  -m $m -e "$d $m $juego- cidetic" -v -o $DIR > $output 2>&1 &
		echo "Running $m on $d for $juego. Logs at: $output"
	done
done
