#!/bin/bash

subj_id=99
B=10
penalty=3
add_query_indicator=1
add_condition_label=0
gating_type=post
n_hidden=32
lr=1e-3
cmpt=1
eta=.1
n_epochs=10
sup_epoch=0
test_mode=1

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
