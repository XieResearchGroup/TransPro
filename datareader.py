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


class Ic_50_Dataset(Dataset):

    def __init__(self, drug_file, data_file, filter, device, cell_ge_file_name):
        super(Ic_50_Dataset, self).__init__()
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

class Ic_50_DataLoader(pl.LightningDataModule):

    def __init__(self, drug_file, Ic_50_file_train,Ic_50_file_dev, Ic_50_file_test,
                 filter, device, cell_ge_file_name, batch_size = 32):
        super(Ic_50_DataLoader, self).__init__()
        self.batch_size = batch_size
        self.train_data_file = Ic_50_file_train
        self.dev_data_file = Ic_50_file_dev
        self.test_data_file = Ic_50_file_test
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


if __name__ == '__main__':
    filter = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422'], "pert_type": ["trt_cp"],
              "cell_id": ["A375", "HT29", "MCF7", "PC3", "HA1E", "YAPC", "HELA"],
              "pert_idose": ["0.04 um", "0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}
              
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
