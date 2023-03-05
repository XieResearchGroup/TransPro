#Train deepNN models for adr prediction using 
#a. Experimental transcriptomics 
# --gx_file "data/Side_effect/${cur_dataset}_PTs_allCells_${conf_level}conf_CellLineDGX.csv"
#b. Experimental proteomics
# --exp_pros 1 
# --gx_file "data/Side_effect/Prot_${cur_dataset}_experimental.csv"
#c. TransPro output proteomics.
# --gx_file "data/side_effect/0423Pros_preds_512ab_${cur_dataset}_PTs_allCells_${conf_level}conf_3.33um_w_pertTrans.csv" 

for cur_dataset in SIDERS FAERS 
do
for conf_level in 0.3 0.5
do
seed=66
python adr_prediction/deepNN_adr_prediction.py \
--cur_dataset ${cur_dataset}  --conf_level ${conf_level} --exp_pros 0 \
--gx_file "data/side_effect/0423Pros_preds_512ab_${cur_dataset}_PTs_allCells_${conf_level}conf_3.33um_w_pertTrans.csv"  \
--seed ${seed} --device 5 >2023_logs/deepNN_${cur_dataset}_${seed}_${state}_${conf_level}conf.log 2>&1 &
done 
done
