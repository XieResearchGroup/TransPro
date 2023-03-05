#!/usr/bin/env bash
#for split in stadScaf cell_split_0 cell_split_1 cell_split_2 rand_split_0 rand_split_1 rand_split_2
#do
#for seed in 27 72 343 
#do 
exp_id=0304_train_reproduce
seed=343
split=cell_split_0
python transPro.py \
--exp_id ${exp_id} --dop 0.2 --seed ${seed} --infer_mode 0 \
--warmup_epochs 600 --max_epochs 1500 --lr_low 0.0001 --lr_high 0.0002 \
--include_trans 1 --use_transmitter 1 --device 2 \
--pert_trans_train_dir "data/adjusted_l1000_pert_trans_full.csv" \
--pert_trans_dev_dir "data/adjusted_l1000_pert_trans_dev.csv" \
--pert_trans_test_dir "data/adjusted_l1000_pert_trans_test.csv" \
--pert_pros_train_dir "data/${split}_512ab_noImpu_pert_pros_train.csv" \
--pert_pros_dev_dir "data/${split}_512ab_pert_pros_dev.csv" \
--pert_pros_test_dir "data/${split}_512ab_pert_pros_test.csv" \
--drug_file_dir "data/a_gdsc_drugs_smiles_pro.csv" \
--trans_basal_dir "data/CCLE_x1305_978genes.csv" \
> 2023_logs/${exp_id}_seed_${seed}.log 2>&1 &
#done
#done
