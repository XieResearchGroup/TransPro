
for split in rand_0 #rand_1 rand_2 rand_3
do
for lr in 0.00005 #0.00002 0.0001  0.00005 
do
for dop in 0.2
do
batch_size=128
job=perturbed_pros # perturbed_trans
seed=343
exp_id=${job}_emb_${split}_bs_${batch_size}_dop_${dop}_lr_${lr}_sd_${seed}
python transPro_Ic50_ft.py \
--exp_id ${exp_id} --dop ${dop} --seed ${seed} --batch_size ${batch_size}  \
--device 5 --max_epochs 5 --lr ${lr} --job ${job} \
--Ic_50_train_dir "data/Ic50/${split}_train_ic50_wo_na.csv" \
--Ic_50_dev_dir "data/Ic50/${split}_dev_ic50_wo_na.csv" \
--Ic_50_test_dir "data/Ic50/rand_0123_test_ic50_wo_na.csv" \
--drug_file_dir "data/a_gdsc_drugs_smiles_pro.csv" \
--trans_basal_dir "data/CCLE_x1305_978genes.csv" \
--pretrained_model_dir "data/trained_model/final_transPro_model.pt" \
> 2023_logs/${exp_id}.log 2>&1 &
done
done
done