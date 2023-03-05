import numpy as np
import random
import torch
#from molecules import Molecules
import pdb
import pandas as pd
from chem_loader import *

def read_drug_string(input_file):
    with open(input_file, 'r') as f:
        drug = dict()
        for line in f:
            line = line.strip().split(',')
            assert len(line) == 2, "Wrong format"
            drug[line[0]] = line[1]
    return drug, None

def choose_mean_example(examples):
    num_example = len(examples)
    mean_value = (num_example - 1) / 2
    indexes = np.argsort(examples, axis=0)
    indexes = np.argsort(indexes, axis=0)
    indexes = np.mean(indexes, axis=1)
    distance = (indexes - mean_value)**2
    index = np.argmin(distance)
    return examples[index]

def read_data(input_file, filter=None):
    """
    :param input_file: including the time, pertid, perttype, cellid, dosage and the perturbed gene expression file (label)
    :param filter: help to check whether the pertid is in the research scope, cells in the research scope ...
    :return: the features, labels and cell type
    """
    feature = []
    label = []
    data = dict()
    pert_id = []
    with open(input_file, 'r') as f:
        f.readline()  # skip header
        for line in f:
            line = line.strip().split(',')
            # assert len(line) == 983 or len(line) == 7 or len(line) == 6, "Wrong format"
            # antibody: 549, rest: 977
            if filter is None:
                ft = ','.join(line[:5])
                data[ft]='foo'
            else:
                # if filter["time"] in line[0] and line[1] not in filter['pert_id'] and line[2] in filter["pert_type"] \
                #     and line[3] in filter['cell_id'] and line[4] in filter["pert_idose"]:
                ft = ','.join(line[:5])
                # print(ft)
                lb = [float(i) for i in line[5:]]
                if ft in data.keys():
                    data[ft].append(lb)
                else:
                    data[ft] = [lb]

    if filter is not None:
        for ft, lb in sorted(data.items()):
            ft = ft.split(',')
            feature.append(ft)
            pert_id.append(ft[1])
            if len(lb) == 1:
                label.append(lb[0])
            else:
                lb = choose_mean_example(lb)
                label.append(lb)
        _, cell_type = np.unique(np.asarray([x[3] for x in feature]), return_inverse=True)
        return np.asarray(feature), np.asarray(label, dtype=np.float64), cell_type
    else:
        for ft, lb in sorted(data.items()):
            ft = ft.split(',')
            feature.append(ft)
            pert_id.append(ft[1])
        return np.asarray(feature)

def transform_to_tensor_per_dataset(feature,  drug,device, basal_expression_file, label=None,):

    """
    :param feature: features like pertid, dosage, cell id, etc. will be used to transfer to tensor over here
    :param label:
    :param drug: ??? a drug dictionary mapping drug name into smile strings
    :param device: save on gpu device if necessary
    :return:
    """
    if not basal_expression_file.endswith('csv'):
        basal_expression_file += '.csv'
    basal_cell_line_expression_feature_csv = pd.read_csv(basal_expression_file, index_col = 0)
    drug_feature = []
    drug_target_feature = []
    pert_type_set = sorted(list(set(feature[:, 2])))
    cell_id_set = sorted(list(set(feature[:,3])))
    pert_idose_set = sorted(list(set(feature[:, 4])))
    # pert_type_set = ['trt_cp']
    # cell_id_set = ['HA1E', 'HT29', 'MCF7', 'YAPC', 'HELA', 'PC3', 'A375']
    # pert_idose_set = ['1.11 um', '0.37 um', '10.0 um', '0.04 um', '3.33 um', '0.12 um']
    use_pert_type = False
    use_cell_id = True ## cell feature will always used
    use_pert_idose = False
    if len(pert_type_set) > 1:
        pert_type_dict = dict(zip(pert_type_set, list(range(len(pert_type_set)))))
        final_pert_type_feature = []
        use_pert_type = True
  
    cell_id_dict = dict(zip(cell_id_set, list(range(len(cell_id_set)))))
    final_cell_id_feature = []
    use_cell_id = True
    if len(pert_idose_set) > 1:
        pert_idose_dict = dict(zip(pert_idose_set, list(range(len(pert_idose_set)))))
        final_pert_idose_feature = []
        use_pert_idose = True

    for i, ft in enumerate(feature):
        drug_fp = drug[ft[1]]
        drug_fp = AllChem.MolFromSmiles(drug_fp)
        drug_fp = mol_to_graph_data_obj_simple(drug_fp)
        drug_feature.append(drug_fp)
        if use_pert_type:
            pert_type_feature = np.zeros(len(pert_type_set))
            pert_type_feature[pert_type_dict[ft[2]]] = 1
            final_pert_type_feature.append(np.array(pert_type_feature, dtype=np.float64))
        if use_cell_id:
            cell_id_feature = basal_cell_line_expression_feature_csv.loc[ft[3],:] ## new_code
            final_cell_id_feature.append(np.array(cell_id_feature, dtype=np.float64))
        if use_pert_idose:
            pert_idose_feature = np.zeros(len(pert_idose_set))
            pert_idose_feature[pert_idose_dict[ft[4]]] = 1
            final_pert_idose_feature.append(np.array(pert_idose_feature, dtype=np.float64))

    feature_dict = dict()
    #feature_dict['drug'] = np.asarray(drug_feature)
    feature_dict['drug'] = drug_feature
    if use_pert_type:
        feature_dict['pert_type'] = torch.from_numpy(np.asarray(final_pert_type_feature, dtype=np.float64)).to(device)
    if use_cell_id:
        feature_dict['cell_id'] = torch.from_numpy(np.asarray(final_cell_id_feature, dtype=np.float64)).to(device)
    if use_pert_idose:
        feature_dict['pert_idose'] = torch.from_numpy(np.asarray(final_pert_idose_feature, dtype=np.float64)).to(device)
    if label is None:
        return feature_dict, use_pert_type, use_cell_id, use_pert_idose 
   
    label_regression = torch.from_numpy(label).to(device)
    return feature_dict, label_regression, use_pert_type, use_cell_id, use_pert_idose

if __name__ == '__main__':
    filter = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422'], "pert_type": ["trt_cp"],
              "cell_id": ['A375', 'HA1E', 'HELA', 'HT29', 'MCF7', 'PC3', 'YAPC'],
              "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}
    ft, lb = read_data('../data/signature_train.csv', filter)
    print(np.shape(ft))
    print(np.shape(lb))
