#!/bin/bash

juego=juego1
NOW=$(date +"%Y%m%d_%H%M%S_%N")
DIR=experiments/syn/$juego/$NOW

python3 syn_generator.py -d enron -l 53 -s streams/enron_syn_1702_skew1_ld75.arff streams/enron_1702_rbf.arff streams/enron_100000_rbf.arff -S MOA_1K JC_1K JC_100K -e "$juego" -o "$DIR/enron"

python3 syn_generator.py -d 20ng -l 20 -s streams/20ng_syn_19300_skew0_ld1.arff streams/20ng_19300_rbf.arff streams/20ng_80000_rbf.arff -S MOA_19K JC_19K JC_80K -e "$juego" -o "$DIR/20ng"

python3 syn_generator.py -d mediamill -l 101 -s streams/mediamill_syn_43907_skew1_ld75.arff streams/mediamill_43907_rbf.arff streams/mediamill_500000_rbf.arff -S MC_MOA MC_JC_43K MC_JC_500K -e "$juego" -o "$DIR/mediamill"
