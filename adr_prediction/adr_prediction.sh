'''
python adr_prediction.py : for scaffold split on the side effect prediction with all the drugs available in the experimental
transcriptomics data;
python special_adr_prediction.py : for testing on the side effect prediction with the shared drugs
between the experimental transcriptomics and proteomics data.
'''

cur_dataset="SIDERS"
split_method="rand_scaffold_split"
for conf_level in 0.3 0.5 
do
state="Pros"
for seed in 66   # shared drugs 97 777 7777 # rand scaf faers 29 34 39 siders 20 44 66 
do
python adr_prediction.py \
--cur_dataset ${cur_dataset} --split_method ${split_method} \
--gx_file "data/side_effect/0423${state}_preds_512ab_${cur_dataset}_PTs_allCells_${conf_level}conf_3.33um_w_pertTrans.csv"  \
--fold 8 --seed ${seed} --device 1 >../Apr_logs/deepNN_${cur_dataset}_${split_method}_${seed}_${state}_${conf_level}conf.log 2>&1 &
done 
done
