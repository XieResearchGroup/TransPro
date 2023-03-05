#!/usr/bin/env bash
exp_id=0301_reproduce_get_final_prediction
python transPro.py \
--exp_id ${exp_id}  --infer_mode 2 \
--device 5   --include_trans 0 --use_transmitter 1 \
--pretrained_model_dir "data/trained_model/final_transPro_model.pt" \
--predicted_result_for_testset "predictions/result_${exp_id}.csv" \
--pert_pros_test_dir "data/1024_MCF7_single_target_drugs_to_predictx656.csv" \
>2023_logs/${exp_id}.log 2>&1 &

