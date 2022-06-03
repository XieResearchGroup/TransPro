import numpy as np
import random
import torch
import data_utils
import pandas as pd
import pdb
from torch.utils.data import random_split, DataLoader, Dataset, ConcatDataset
import pytorch_lightning as pl
import torch_geometric as tg
from chem_loader import *


class PretrainTransProsDataset(Dataset):

    '''
    This dataset can be used to prepare for trans_only dataloader, pros_only dataloader and trans_pros dataloader (intersection)
    '''

    def __init__(self, feature_data_dir, label_data_dir):

        '''
        Args:
            feature_data_dir (string): path to the feature data file
            label_data_dir (string): path to the label data file
        '''
        super(PretrainTransProsDataset, self).__init__()
        self._pretrain_feature_data = pd.read_csv(feature_data_dir, index_col = 0)
        self._pretrain_label_data = pd.read_csv(label_data_dir, index_col = 0)
        assert sum(self._pretrain_feature_data.index == self._pretrain_label_data.index) == len(
            self._pretrain_label_data), "the feature and label data are not matched"

    def __len__(self):
        return len(self._pretrain_feature_data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self._pretrain_feature_data.iloc[idx, :].values, self._pretrain_label_data.iloc[idx, :].values

class PretrainTransProsDataloader(pl.LightningDataModule):

    '''
    This dataloader is used for the trans_only dataloader, pros_only dataloader and trans_pros dataloader (intersection)
    '''
    def __init__(self, feature_data_dir, label_data_dir, batch_size = 32):

        '''
        Args:
            feature_data_dir (string): path to the feature data file
            label_data_dir (string): path to the label data file
            batch_size (int): batch size
        '''
        super(PretrainTransProsDataloader, self).__init__()
        self.feature_data_dir = feature_data_dir
        self.label_data_dir = label_data_dir
        self.batch_size = batch_size

    def setup(self, ):
        '''
        split data to train, val and test randomly
        '''
        random.seed(42)
        self.full_dataset = PretrainTransProsDataset(self.feature_data_dir, self.label_data_dir)
        full_dataset_len = len(self.full_dataset)
        self.train_len, self.val_len, self.test_len = self._get_train_val_test_len(full_dataset_len)
        train_val_data, self.test_data = random_split(self.full_dataset, [self.train_len+self.val_len, self.test_len])
        self.train_data, self.val_data = random_split(train_val_data, [self.train_len, self.val_len])
    
    def train_dataloader(self):

        return DataLoader(self.train_data, batch_size = self.batch_size)
    
    def val_dataloader(self):

        return DataLoader(self.val_data, batch_size = self.batch_size)

    def test_dataloader(self):

        return DataLoader(self.test_data, batch_size = self.batch_size)
    
    def full_dataloader(self):
        '''
        This is used to do pretraining with all dataset
        '''
        return DataLoader(self.full_dataset, batch_size = self.batch_size)


    def _get_train_val_test_len(self, full_data_len):

        train_len = full_data_len // 5 * 3
        test_len = full_data_len // 5
        val_len = full_data_len - train_len - test_len
        return train_len, val_len, test_len

class PerturbedDataset(Dataset):

    def __init__(self, drug_file, data_file, filter, device, cell_ge_file_name):
        super(PerturbedDataset, self).__init__()
        self.device = device
        self.filter = filter
        
        self.drug, self.drug_dim = data_utils.read_drug_string(drug_file)
        if self.filter is None:
            feature = data_utils.read_data(data_file)
            self.feature, self.use_pert_type, self.use_cell_id, self.use_pert_idose = \
            data_utils.transform_to_tensor_per_dataset(feature, self.drug, self.device, cell_ge_file_name)
        else:
            feature, label, self.cell_type = data_utils.read_data(data_file, filter)
            self.feature, self.label, self.use_pert_type, self.use_cell_id, self.use_pert_idose = \
                data_utils.transform_to_tensor_per_dataset(feature, self.drug, self.device, cell_ge_file_name, label)

    def __len__(self):
        #return self.feature['drug'].shape[0]
        return len(self.feature['drug'])

    def __getitem__(self, idx):
        output = dict()
        output['drug'] = self.feature['drug'][idx]
        if self.use_pert_type:
            output['pert_type'] = self.feature['pert_type'][idx]
        if self.use_cell_id:
            output['cell_id'] = self.feature['cell_id'][idx]
        if self.use_pert_idose:
            output['pert_idose'] = self.feature['pert_idose'][idx]
        if self.filter is None:
            return  output
        return output, self.label[idx], self.cell_type[idx]

class PerturbedDataLoader(pl.LightningDataModule):

    def __init__(self, drug_file, data_file_train, data_file_dev, data_file_test,
                 filter, device, cell_ge_file_name, batch_size = 32):
        super(PerturbedDataLoader, self).__init__()
        self.batch_size = batch_size
        self.train_data_file = data_file_train
        self.dev_data_file = data_file_dev
        self.test_data_file = data_file_test
        self.drug_file = drug_file
        self.filter = filter
        self.device = device
        self.cell_ge_file_name = cell_ge_file_name
    
    def collate_fn(self, batch):
        features = {}
        # features['drug'] = data_utils.convert_smile_to_feature([output['drug'] for output, _, _ in batch], self.device)
        # features['mask'] = data_utils.create_mask_feature(features['drug'], self.device)
        if self.filter is  None:
            features['drug'] = [output['drug'] for output in batch]
            chem_loader = tg.loader.DataLoader(features['drug'],batch_size=len(batch),shuffle=False)
            for chem_batch in chem_loader:
                features['drug'] = chem_batch
            for key in batch[0].keys():
                if key == 'drug':
                    continue
                features[key] = torch.stack([output[key] for output in batch], dim = 0)
            return features

        else:
            features['drug'] = [output['drug'] for output, _, _ in batch]
            chem_loader = tg.loader.DataLoader(features['drug'],batch_size=len(batch),shuffle=False)
            for chem_batch in chem_loader:
                features['drug'] = chem_batch
            for key in batch[0][0].keys():
                if key == 'drug':
                    continue
                features[key] = torch.stack([output[key] for output, _, _ in batch], dim = 0)
            labels = torch.stack([label for _, label, _ in batch], dim = 0)
            cell_types = torch.Tensor([cell_type for _, _, cell_type in batch])
            return features, labels, torch.Tensor(cell_types).to(self.device)

    def prepare_data(self):
        '''
        Use this method to do things that might write to disk or that need to be \
            done only from a single GPU in distributed settings.
        how to download(), tokenize, the processed file need to be saved to disk to be accessed by other processes
        prepare_data is called from a single GPU. Do not use it to assign state (self.x = y).
        '''
        pass

    def setup(self, stage = None):
        self.train_data = PerturbedDataset(self.drug_file, self.train_data_file,
                 self.filter, self.device, self.cell_ge_file_name)
        self.dev_data = PerturbedDataset(self.drug_file, self.dev_data_file,
                 self.filter, self.device, self.cell_ge_file_name)
        self.test_data = PerturbedDataset(self.drug_file, self.test_data_file,
                 self.filter, self.device, self.cell_ge_file_name)
        self.use_pert_type = self.train_data.use_pert_type
        self.use_cell_id = self.train_data.use_cell_id
        self.use_pert_idose = self.train_data.use_pert_idose
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.batch_size, shuffle = True, collate_fn = self.collate_fn,drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.dev_data, batch_size = self.batch_size, collate_fn = self.collate_fn,drop_last=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size = self.batch_size, collate_fn = self.collate_fn,drop_last=False)
    
    def full_dataloader(self):
        return DataLoader(self.full_data, batch_size = self.batch_size, collate_fn = self.collate_fn,drop_last=False)

class DataReader(object):

    def __init__(self, drug_file, gene_file, data_file_train, data_file_dev, data_file_test,
                 filter, device, cell_ge_file_name):
        self.device = device
        self.drug, self.drug_dim = data_utils.read_drug_string(drug_file)
        self.gene = data_utils.read_gene(gene_file, self.device)
        feature_train, label_train, self.train_cell_type = data_utils.read_data(data_file_train, filter)
        feature_dev, label_dev, self.dev_cell_type = data_utils.read_data(data_file_dev, filter)
        feature_test, label_test, self.test_cell_type = data_utils.read_data(data_file_test, filter)
        self.train_feature, self.dev_feature, self.test_feature, self.train_label, \
        self.dev_label, self.test_label, self.use_pert_type, self.use_cell_id, self.use_pert_idose = \
            data_utils.transfrom_to_tensor(feature_train, label_train, feature_dev, label_dev,
                                           feature_test, label_test, self.drug, self.device, cell_ge_file_name)

    def get_batch_data(self, dataset, batch_size, shuffle):
        if dataset == 'train':
            feature = self.train_feature
            label = self.train_label
            cell_type = torch.Tensor(self.train_cell_type).to(self.device)
        elif dataset == 'dev':
            feature = self.dev_feature
            label = self.dev_label
            cell_type = torch.Tensor(self.dev_cell_type).to(self.device)
        elif dataset == 'test':
            feature = self.test_feature
            label = self.test_label
            cell_type = torch.Tensor(self.test_cell_type).to(self.device)
        if shuffle:
            index = torch.randperm(len(feature['drug'])).long()
            index = index.numpy()
        for start_idx in range(0, len(feature['drug']), batch_size):
            if shuffle:
                excerpt = index[start_idx: start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            output = dict()
            output['drug'] = data_utils.convert_smile_to_feature(feature['drug'][excerpt], self.device)
            output['mask'] = data_utils.create_mask_feature(output['drug'], self.device)
            if self.use_pert_type:
                output['pert_type'] = feature['pert_type'][excerpt]
            if self.use_cell_id:
                output['cell_id'] = feature['cell_id'][excerpt]
            if self.use_pert_idose:
                output['pert_idose'] = feature['pert_idose'][excerpt]
            yield output, label[excerpt], cell_type[excerpt]


if __name__ == '__main__':
    filter = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422'], "pert_type": ["trt_cp"],
              "cell_id": ["A375", "HT29", "MCF7", "PC3", "HA1E", "YAPC", "HELA"],
              "pert_idose": ["0.04 um", "0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}
              
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
