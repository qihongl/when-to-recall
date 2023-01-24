#!/bin/bash

add_query_indicator=1
add_condition_label=0
gating_type=post
n_epochs=15000
sup_epoch=0
test_mode=1

for subj_id in {0..3}
do
  for lr in 1e-3
  do
    for B in 10 16
    do
      for penalty in 1 4 16
      do
        for n_hidden in 128
        do
          for cmpt in .1 .5 1
          do
            for eta in .1
            do
            sbatch train.sh \
                 ${subj_id} \
                 ${B} \
                 ${penalty} \
                 ${add_query_indicator} \
                 ${add_condition_label} \
                 ${gating_type} \
                 ${n_hidden} \
                 ${lr} \
                 ${cmpt} \
                 ${eta} \
                 ${n_epochs} \
                 ${sup_epoch} \
                 ${test_mode}
            done
          done
        done
      done
    done
  done
done
