#!/usr/bin/env bash
# get embeddings for the adr downstream task.
# for dataset in FAERS SIDERS
# do
# for conf_level in 0.3 0.5
# do 
dataset=FAERS 
conf_level=0.5 
exp_id=0301_reproduce_get_embeddings
python transPro.py \
--exp_id ${exp_id}  --infer_mode 1 \
--device 5   --include_trans 0  --use_transmitter 1 \
--pretrained_model_dir "data/trained_model/final_transPro_model.pt" \
--predicted_result_for_testset "predictions/${dataset}_embeddings.csv" \
--pert_pros_test_dir "data/Side_effect/${dataset}_PTs_allCells_${conf_level}conf_CellLineDGX.csv" \
>2023_logs/${exp_id}.log 2>&1 &
# done 
# done
