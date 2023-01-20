#!/bin/bash
#SBATCH -t 11:55:00
#SBATCH -c 1
#SBATCH --mem-per-cpu 12G

#SBATCH --job-name=cody-exp-2
#SBATCH --output slurm_log/%j.log

LOGROOT=/tigress/qlu/logs/when-to-recall/log
DT=$(date +%Y-%m-%d)

echo $(date)

srun python -u run_exp2.py \
    --subj_id ${1} \
    --B ${2} \
    --penalty ${3} \
    --add_query_indicator ${4} \
    --add_condition_label ${5} \
    --gating_type ${6} \
    --n_hidden ${7} \
    --lr ${8} \
    --cmpt ${9} \
    --eta ${10} \
    --n_epochs ${11} \
    --sup_epoch ${12} \
    --test_mode ${13} \
    --exp_name $DT \
    --log_root $LOGROOT

echo $(date)
