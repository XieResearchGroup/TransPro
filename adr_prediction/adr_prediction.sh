cur_dataset="SIDERS"
split_method="rand_scaffold_split"
for conf_level in 0.3 0.5 
do
state="Pros"
for seed in 66   # shared drugs 97 777 7777 # rand scaf faers 29 34 39 siders 20 44 66 
do
python adr_prediction.py \
--cur_dataset ${cur_dataset} --split_method ${split_method} \
--gx_file "/raid/home/yoyowu/PertPro/perturbed_proteomics/data/side_effect/0423${state}_preds_512ab_${cur_dataset}_PTs_allCells_${conf_level}conf_3.33um_w_pertTrans.csv"  \
--fold 8 --seed ${seed} --device 1 >../Apr_logs/deepNN_${cur_dataset}_${split_method}_${seed}_${state}_${conf_level}conf.log 2>&1 &
done 
done
#/raid/home/yoyowu/MultiDCP/MultiDCP_data/side_effect/FAERS_offsides_PTs.csv
#/raid/home/yoyowu/PertPro/perturbed_proteomics/data/side_effect/FAERS_PTs_allCells_0.3conf_3.33um_CellLineDGX_x2142.csv
#2192
#/raid/home/yoyowu/PertPro/perturbed_proteomics/data/side_effect/Trans_preds_FAERS_PTs_allCells_0.3conf_3.33um_w_ATTNgin.csv
# /raid/home/yoyowu/PertPro/perturbed_proteomics/data/side_effect/${state}_preds_${cur_dataset}_PTs_allCells_${conf_level}conf_3.33um_w_ATTNgin.csv

#/raid/home/yoyowu/MultiDCP/MultiDCP_data/side_effect/SIDER_PTs.csv
#/raid/home/yoyowu/PertPro/perturbed_proteomics/data/side_effect/SIDERS_PTs_allCells_0.3conf_3.33um_CellLineDGX_x2274.csv
#/raid/home/yoyowu/PertPro/perturbed_proteomics/data/side_effect/SIDERS_PTs_allCells_0.5conf_3.33um_CellLineDGX_x2331.csv
#/raid/home/yoyowu/PertPro/perturbed_proteomics/data/side_effect/Pros_preds_SIDERS_PTs_allCells_0.3conf_3.33um_w_ATTNgin.csv

#"/raid/home/yoyowu/PertPro/perturbed_proteomics/data/side_effect/${state}_preds_512ab_${cur_dataset}_PTs_allCells_${conf_level}conf_3.33um_w_ginATTN.csv" \

#/raid/home/yoyowu/PertPro/perturbed_proteomics/data/side_effect/0423Pros_preds_512ab_SIDERS_PTs_allCells_0.3conf_3.33um_w_pertTrans.csv
#/raid/home/yoyowu/PertPro/perturbed_proteomics/data/side_effect/0423Pros_preds_512ab_SIDERS_PTs_allCells_0.3conf_3.33um_w_prosOnly.csv
#/raid/home/yoyowu/PertPro/perturbed_proteomics/data/side_effect/0423Pros_preds_512ab_SIDERS_PTs_allCells_0.5conf_3.33um_w_pertTrans.csv
#/raid/home/yoyowu/PertPro/perturbed_proteomics/data/side_effect/0423Pros_preds_512ab_SIDERS_PTs_allCells_0.5conf_3.33um_w_prosOnly.csv