#!/usr/bin/env bash
#split=stadScaf  #cell_split_4 cell_split_2 cell_cluster_1s9 stadScaf rand_split_119 cell_split_1 cell_split_2 
#for split in stadScaf
#do
#for seed in 27 72 
#do 
exp_id=0422_get_pertTrans_w_transmitter_model
split=cell_split_1
python ../perturbed_proteomics/transPro.py \
--exp_id ${exp_id} --dop 0.2 --seed 343 \
--device 0  --warmup_epochs 600 --max_epochs 1500 --lr_low 0.0001 --lr_high 0.0002 --include_trans 1 --use_transmitter 1 \
--pert_trans_train_dir "data/adjusted_l1000_pert_trans_full.csv" \
--pert_trans_dev_dir "data/adjusted_l1000_pert_trans_dev.csv" \
--pert_trans_test_dir "data/adjusted_l1000_pert_trans_test.csv" \
--pert_pros_train_dir "data/${split}_512ab_noImpu_pert_pros_dev.csv" \
--pert_pros_dev_dir "data/${split}_512ab_pert_pros_dev.csv" \
--pert_pros_test_dir "data/${split}_512ab_pert_pros_test.csv" \
--drug_file_dir "data/a_gdsc_drugs_smiles_pro.csv" \
--trans_basal_dir "data/CCLE_x1305_978genes.csv" \
--saved_model_path "/raid/home/yoyowu/PertPro/models_inventory/${exp_id}.pt" \
>../Apr_logs/${exp_id}_mixlr_seed_${seed}.log 2>&1 &
#done
#done
