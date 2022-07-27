#!/usr/bin/bash

rdir=`pwd`
PYTHONPATH=`pwd`
SEPERATELINE="==================================\n\n"
PRINTRESULTS="\nPrinting results\n\n"

printf "Note: this script runs split 0 of the 10-fold cross validation as a demonstration. The results may be slightly different from the values reported in the main text.\n\n"

# Run single task model on logp toy dataset

echo
echo "Running single task model to predict noisy logp"
printf $SEPERATELINE

noises="0.00 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28 2.56 5.12"
splits="0 1 2 3 4 5 6 7 8 9"

for noise in $noises;
do
    wdir=results/logp/$noise/;
    mkdir -p $wdir;
    cd $wdir;
    echo "Running model at $wdir" 
    CUDA_VISIBLE_DEVICES=0 python $rdir/single_task_train.py --lr 0.001 --batch-size 16 --epochs 200 --fea-len 16 --n-layers 4 --n-h 2 --split 0 --has-h 0 --form-ring 1 --log10 0 $rdir/data/logp/noise_$noise > hp.out;
    cd $rdir;
done

printf $PRINTRESULTS

python postprocess.py logp results/logp/

# Run single task model on the noisy 5 ns conductivity

echo
echo "Running single task model to predict noisy 5ns conductivity"
printf $SEPERATELINE

wdir=results/cond_rand;
mkdir -p $wdir;
cd $wdir;
echo "Running model at $wdir"
CUDA_VISIBLE_DEVICES=0 python $rdir/single_task_train.py --lr 0.001 --batch-size 16 --epochs 200 --fea-len 16 --n-layers 4 --n-h 1 --split 0 --has-h 0 --form-ring 1 $rdir/data/conductivity/5ns > hp.out;
cd $rdir;

printf $PRINTRESULTS
python postprocess.py poly_rand results/cond_rand/


# Run multi-task model for property interpolation

echo
echo "Running multi-task model to reduce systematic errors"
printf $SEPERATELINE

poly_props="conductivity li_diff tfsi_diff poly_diff"

echo
echo "Running model for interpolation performace"
echo

for poly_prop in $poly_props;
do
    wdir=results/systematic_int/$poly_prop;
    mkdir -p $wdir;
    cd $wdir;
    echo "Running model at $wdir" 
    CUDA_VISIBLE_DEVICES=0 python $rdir/multi_task_train.py --lr 0.001 --batch-size 8 --epochs 200 --fea-len 16 --n-layers 4 --n-h 2 --split 0 --has-h 0 --form-ring 1 --exp-weight 0.1 --use-sim-pred 1 $rdir/data/conductivity/5ns $rdir/data/conductivity/50ns > hp.out;
    cd $rdir;
done

printf $PRINTRESULTS
python postprocess.py poly_sys results/systematic_int/

echo
echo "Running model for extrapolation performace"
echo

for poly_prop in $poly_props;
do
    wdir=results/systematic_ext/$poly_prop;
    mkdir -p $wdir;
    cd $wdir;
    echo "Running model at $wdir" 
    CUDA_VISIBLE_DEVICES=0 python $rdir/multi_task_train.py --lr 0.001 --batch-size 8 --epochs 200 --fea-len 16 --n-layers 4 --n-h 2 --split 0 --has-h 0 --form-ring 1 --exp-weight 0.1 --use-sim-pred 1 $rdir/data/conductivity/5ns $rdir/data/conductivity/50ns_extrapolate > hp.out;
    cd $rdir;
done

printf $PRINTRESULTS
python postprocess.py poly_sys results/systematic_ext
