import pandas as pd
import random
from sklearn.model_selection import KFold, GroupKFold
import pdb
import numpy as np
from itertools import compress,  chain
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
import torch



class ADRDataBuilder:

    def __init__(self, file_name):
        '''
        build the side effect dataframe from either FAERS_offsides_PTs.csv or *_PTs.csv
        :file_name: str
        '''
        self.side_effect_df = pd.read_csv(file_name, index_col = 0)
        #self.og_se = self.side_effect_df 
        # self.__remove_drugs_with_less_ADR()
        self.__remove_adrs_with_less_drugs()
        assert self.side_effect_df.shape[0] == len(set(self.side_effect_df.index)), "the pert id has duplications"
    
    def __remove_adrs_with_less_drugs(self):
        '''
        remove the adrs with very few drugs
        '''
        new_df = self.side_effect_df[self.get_side_effect_names()]
        sparse_filter = new_df.values.sum(axis = 0) > 10
        #sparse_filter = new_df.values.sum(axis = 0) > 0
        self.side_effect_df = self.side_effect_df.loc[: ,sparse_filter]

    def __remove_drugs_with_less_ADR(self):
        '''
        remove the drugs with very few ADRs
        '''
        new_df = self.side_effect_df[self.get_side_effect_names()]
        sparse_filter = new_df.values.sum(axis = 1) > 30
        self.side_effect_df = self.side_effect_df.loc[sparse_filter,:]

    def get_whole_df(self):
        return self.side_effect_df

    def get_side_effects_df_only(self):
        '''
        the return columns only have the differetn side effects
        '''
        return self.side_effect_df[self.get_side_effect_names()]

    def get_side_effect_names(self):
        cols = list(self.side_effect_df.columns)
        if 'pert_id' in cols:
            cols.remove('pert_id')
        return cols

    def get_drug_list(self):
        return list(self.side_effect_df.index)

    def prepare_adr_df_basedon_perts(self, pertid_list):
        '''
        prepare the side effect profile based on the pert id list
        :pertid_list: list
        :return: dataframe: the dataframe with index as pertid_list
        '''
        extra_pert = set(pertid_list) - set(self.get_drug_list())
        assert len(extra_pert) == 0, "there are pertid not found in the ADR file"
        return_cols = self.get_side_effect_names()
        return self.side_effect_df.loc[pertid_list, return_cols]

class PerturbedDGXDataBuilder:

    def __init__(self, gx_file_name, pert_list, pred_flag=False, cs_part = True):
        '''
        build the pertubed gene expression dataframe from either FAERS_offsides_PTs_PredictionDGX.csv or *_PTs_PredictionDGX.csv
        :file_name: str
        :pred_flag: Boolean, whether the build dataframe is predicted DGX or groundtruth DGX
        '''
        self._pred_flag = pred_flag
        self.dgx_df = pd.read_csv(gx_file_name)
        #self.cs_df = pd.read_csv(drug_cs_dir, index_col = 0)
        pertid_set = set(pert_list)
        self.x_ls = []
        self.dgx_df = self.dgx_df.loc[self.dgx_df['pert_id'].isin(pertid_set),:]
        self.x_ls.append(self.dgx_df)
        #self.cs_df = self.cs_df.loc[self.dgx_df.pert_id, :]
        #assert len(self.dgx_df) == sum(self.dgx_df.pert_id == self.cs_df.index), "dgx and cs df has different pert_ids "
        if cs_part:
            self.x_ls.append(self.cs_df.reset_index(drop = True))
        self.x_df = pd.concat(self.x_ls, axis = 1)
       
        # self.dgx_df = self.dgx_df.drop_duplicates('pert_id')

    def get_whole_df(self):
        return self.x_df

    def get_pred_flag(self):
        return self._pred_flag

    def get_filter_df(self, pertid_list):
        '''
        :pertid_list: list: a list of pert ids to filter the dataframe
        :return: dataframe: processed dataframe
        '''
        pertid_set = set(pertid_list)
        return self.x_df.loc[self.x_df['pert_id'].isin(pertid_set),:]
    
    def get_pert_id_list(self):
        return list(self.x_df.pert_id)

    def get_gx_only(self):
        '''
        the return colums only the gene expression features
        '''
        return self.x_df.iloc[:, 5:]
    def get_ab_only(self):
        '''
        the return colums only the proteomics expression'''
        return self.x_df.iloc[:,:-1]

class XYPreparer:

    def __init__(self, X, Y, split_list, random_seed):

        self.X = X
        self.Y = Y
        self.split_list = split_list
        self.random_seed = random_seed
        self.drug_file = pd.read_csv('/raid/home/yoyowu/PertPro/perturbed_proteomics/data/drugs_smiles_pro.csv',index_col=0)
        self.drug_dict,_ = self.read_drug_string('/raid/home/yoyowu/PertPro/perturbed_proteomics/data/drugs_smiles_pro.csv')
        self.smiles_all = [self.drug_dict[pert_id] for pert_id in split_list]
    def k_fold_split(self):
        kf = KFold(n_splits = 5, random_state = self.random_seed)
        for train_index, test_index in kf.split(self.X, self.Y):
            yield train_index, test_index

    def leave_new_drug_out_split(self):
        random.seed(self.random_seed)
        gkf = GroupKFold(n_splits = 5)
        for train_index, test_index in gkf.split(self.X, None, self.split_list):
            yield train_index, test_index

    def read_drug_string(self,input_file):
        with open(input_file, 'r') as f:
            drug = dict()
            for line in f:
                line = line.strip().split(',')
                assert len(line) == 2, "Wrong format"
                drug[line[0]] = line[1]
        return drug, None


    def generate_scaffold(self,smiles, include_chirality=False):
            """
            Obtain Bemis-Murcko scaffold from smiles
            :param smiles:
            :param include_chirality:
            :return: smiles of scaffold
            """
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                smiles=smiles, includeChirality=include_chirality)
            return scaffold

        # # test generate_scaffold
        # s = 'Cc1cc(Oc2nccc(CCC)c2)ccc1'
        # scaffold = generate_scaffold(s)
        # assert scaffold == 'c1ccc(Oc2ccccn2)cc1'

    def scaffold_split(self,dataset, smiles_list, task_idx=None, null_value=0,
                    frac_train=0.8, frac_valid=0.0, frac_test=0.2,
                    return_smiles=False):
        """
        Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
        Split dataset by Bemis-Murcko scaffolds
        This function can also ignore examples containing null values for a
        selected task when splitting. Deterministic split
        :param dataset: pytorch geometric dataset obj
        :param smiles_list: list of smiles corresponding to the dataset obj
        :param task_idx: column idx of the data.y tensor. Will filter out
        examples with null value in specified task column of the data.y tensor
        prior to splitting. If None, then no filtering
        :param null_value: float that specifies null value in data.y to filter if
        task_idx is provided
        :param frac_train:
        :param frac_valid:
        :param frac_test:
        :param return_smiles:
        :return: train, valid, test slices of the input dataset obj. If
        return_smiles = True, also returns ([train_smiles_list],
        [valid_smiles_list], [test_smiles_list])
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

        if task_idx != None:
            # filter based on null values in task_idx
            # get task array
            y_task = np.array([data.y[task_idx].item() for data in dataset])
            # boolean array that correspond to non null values
            non_null = y_task != null_value
            smiles_list = list(compress(enumerate(smiles_list), non_null))
        else:
            non_null = np.ones(len(dataset)) == 1
            smiles_list = list(compress(enumerate(smiles_list), non_null))

        # create dict of the form {scaffold_i: [idx1, idx....]}
        all_scaffolds = {}
        for i, smiles in smiles_list:
            scaffold = self.generate_scaffold(smiles, include_chirality=True)
            if scaffold not in all_scaffolds:
                all_scaffolds[scaffold] = [i]
            else:
                all_scaffolds[scaffold].append(i)

        # sort from largest to smallest sets
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]

        # get train, valid test indices
        train_cutoff = frac_train * len(smiles_list)
        valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0


        return train_idx,valid_idx,test_idx


    def random_scaffold_split(self,
        dataset,
        smiles_list,
        task_idx=None,
        null_value=0,
        frac_train=0.7,
        frac_valid=0.0,
        frac_test=0.3,
        seed=7,
    ):
        """
        Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/\
            chainer_chemistry/dataset/splitters/scaffold_splitter.py
        Split dataset by Bemis-Murcko scaffolds
        This function can also ignore examples containing null values for a
        selected task when splitting. Deterministic split
        :param dataset: pytorch geometric dataset obj
        :param smiles_list: list of smiles corresponding to the dataset obj
        :param task_idx: column idx of the data.y tensor. Will filter out
        examples with null value in specified task column of the data.y tensor
        prior to splitting. If None, then no filtering
        :param null_value: float that specifies null value in data.y to filter if
        task_idx is provided
        :param frac_train:
        :param frac_valid:
        :param frac_test:
        :param seed;
        :return: train, valid, test slices of the input dataset obj
        """

        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

        if task_idx is not None:
            # filter based on null values in task_idx
            # get task array
            y_task = np.array([data.y[task_idx].item() for data in dataset])
            # boolean array that correspond to non null values
            non_null = y_task != null_value
            smiles_list = list(compress(enumerate(smiles_list), non_null))
        else:
            non_null = np.ones(len(dataset)) == 1
            smiles_list = list(compress(enumerate(smiles_list), non_null))

        rng = np.random.RandomState(seed)

        scaffolds = defaultdict(list)
        for ind, smiles in smiles_list:
            scaffold = self.generate_scaffold(smiles, include_chirality=True)
            scaffolds[scaffold].append(ind)

        scaffold_sets = rng.permutation(list(scaffolds.values()))

        n_total_valid = int(np.floor(frac_valid * len(dataset)))
        n_total_test = int(np.floor(frac_test * len(dataset)))

        train_idx = []
        valid_idx = []
        test_idx = []

        for scaffold_set in scaffold_sets:
            if len(valid_idx) + len(scaffold_set) <= n_total_valid:
                valid_idx.extend(scaffold_set)
            elif len(test_idx) + len(scaffold_set) <= n_total_test:
                test_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

      

        return train_idx,valid_idx,test_idx

