from abc import ABC, abstractmethod
from matplotlib.pyplot import axis
import torch.nn as nn
import torch
from torch import device, cuda
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import os
import random
import pdb


class ADRPredictionModelABC(ABC):

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def predict_prob(self, X):
        pass

    @abstractmethod
    def fit(self):
        pass

class ADRPredictionDeepModel(nn.Module):
    '''
    The class to build the deep learning model, only the infrastructure
    '''
    def __init__(self, X_dim, Y_dim, layout = [1000,1000,500]):
        '''
        :X_dim: int, input dimension
        :Y_dim: int, output dimenstion
        :layout: list, number of hidden units in each hidden layer
        '''
        super(ADRPredictionDeepModel, self).__init__()
        self.layers = []
        self.Y_dim = Y_dim
        for i, layer_hidden_unit in enumerate([X_dim] + layout):
            if i == len(layout):
                break
            cur_layer = nn.Sequential(nn.Linear(layer_hidden_unit, layout[i]),
                                    nn.BatchNorm1d(layout[i]),
                                    nn.ReLU(),
                                    nn.Dropout(0.4))
            self.layers.append(cur_layer)
        self.MLP = nn.Sequential(*self.layers)
        self.task_specific_list = nn.ModuleList()
        for _ in range(Y_dim):
            output_layer = nn.Sequential(nn.Linear(layout[-1], layout[-1]), 
                                            nn.BatchNorm1d(layout[-1]),
                                            nn.ReLU(),
                                            nn.Linear(layout[-1], 1),
                                            nn.Sigmoid())
            self.task_specific_list.append(output_layer)
        self.initializer = torch.nn.init.kaiming_uniform_
 
    def forward(self, X):
        output = self.MLP(X)
        preds = []
        for i in range(self.Y_dim):
            preds.append(self.task_specific_list[i](output))
        return torch.cat(preds, axis =1)

    def init_weights(self):

        for name, parameter in self.named_parameters():
            if parameter.dim() == 1:
                nn.init.constant_(parameter, 10**-7)
            else:
                self.initializer(parameter)

class DeepModel(ADRPredictionModelABC):

    def __init__(self, X_dim, Y_dim, layout = [1000,1000,500], random_seed = 42, device = device("cpu"), logger = None):
        super(DeepModel, self).__init__()
        self.device = device
        self.model = ADRPredictionDeepModel(X_dim, Y_dim, layout)
        self.model.init_weights()
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00002)
        self.batch_size =128
        self.random_seed = random_seed
        self.epoch = 0
        self.loss = self.prepare_loss()
        self.version = str(split_num)
        self.logger = logger

    def prepare_loss(self):
        cur_loss = nn.BCELoss()
        def new_loss(pred, truth):
            '''
            pred: tensor: batch_size * y_dim
            truth: tensor: batch_size * y_dim
            '''
            losses = 0
            for i in range(pred.size(1)):
                losses += cur_loss(pred[:,i], truth[:,i])
            return losses
        return new_loss

    def get_model(self):
        return self.model

    def predict_prob(self, X):
        self.model.eval()
        Y = []
        with torch.no_grad():
            for i in range(X.shape[0]//self.batch_size + 1):
                new_X = torch.tensor(X[self.batch_size*i:self.batch_size*(i+1), :]).to(self.device).float()
                new_Y = self.model(new_X)
                Y.append(new_Y.cpu().numpy())
        return np.concatenate(Y)
    
    def fit(self, X, Y):
        '''
        :return: float: cur loss values
        '''
        self.model.train()
        index_list = [x for x in range(len(X))]
        random.seed(self.random_seed)
        random.shuffle(index_list)
        loss_ls = []
        shuffled_X = X[index_list,:]
        shuffled_Y = Y[index_list,:]
        Y_pred_ls = []
        finished = False
        for i in range(len(index_list)//self.batch_size + 1):
            if finished:
                break
            self.optimizer.zero_grad()
            cur_idx = index_list[self.batch_size*i:self.batch_size*(i+1)]
            if len(index_list) - self.batch_size*(i+1) < 4:
                cur_idx = index_list[self.batch_size*i:]
                finished = True
            new_X = torch.tensor(X[cur_idx, :]).to(self.device).float()
            new_Y = torch.tensor(Y[cur_idx, :]).to(self.device).float()
            Y_pred = self.model(new_X)
            loss = self.loss(Y_pred, new_Y)
            loss.backward()
            loss_ls.append(loss.item())
            Y_pred_ls.append(Y_pred.detach().cpu().numpy())
            self.optimizer.step()
        shuffled_pred_Y = np.concatenate(Y_pred_ls)
        macro_aucroc = roc_auc_score(shuffled_Y.reshape(-1), shuffled_pred_Y.reshape(-1))
        macro_prauc = average_precision_score(shuffled_Y.reshape(-1), shuffled_pred_Y.reshape(-1))
        micro_aucroc = roc_auc_score(shuffled_Y, shuffled_pred_Y,average="micro")
        micro_prauc = average_precision_score(shuffled_Y, shuffled_pred_Y,average="micro")
        if self.logger is None:
            print("{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5!r}".format(np.mean(loss_ls[:-1]), micro_aucroc, macro_aucroc, micro_prauc, macro_prauc, self.epoch+1))
        else:
            self.logger.debug("{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}, {4:.4f}, {5!r}".format(np.mean(loss_ls[:-1]), micro_aucroc, macro_aucroc, micro_prauc, macro_prauc, self.epoch+1))
        self.epoch += 1
        return np.mean(loss_ls[:-1])
    
    def score(self, X, Y, test_drug_idx):

        self.model.eval()
        Y_pred_ls = []
        with torch.no_grad():
            for i in range(X.shape[0]//self.batch_size + 1):
                new_X = torch.tensor(X[self.batch_size*i:self.batch_size*(i+1), :]).to(self.device).float()
                Y_pred = self.model(new_X)
                Y_pred_ls.append(Y_pred.cpu().numpy())
        
        all_Y_pred = np.concatenate(Y_pred_ls)

        # using the top 1 score of each drug. 
        aucroc1_of_each_drug = []
        prauc1_of_each_drug = []
        for i in range(len(test_drug_idx)):
            drug_spec_Y_pred = np.take(all_Y_pred, test_drug_idx[i],axis=0)
            drug_spec_Y = np.take(Y, test_drug_idx[i],axis=0)
            top_drug_spec_aucroc = max([roc_auc_score(drug_spec_Y[i].reshape(-1), drug_spec_Y_pred[i].reshape(-1)) for i in range(len(drug_spec_Y))])
            top_drug_spec_prauc = max([average_precision_score(drug_spec_Y[i].reshape(-1), drug_spec_Y_pred[i].reshape(-1)) for i in range(len(drug_spec_Y))])
            aucroc1_of_each_drug.append(top_drug_spec_aucroc)
            prauc1_of_each_drug.append(top_drug_spec_prauc)
        avg_of_top_aucroc  = np.mean(aucroc1_of_each_drug)
        avg_of_top_prauc = np.mean(prauc1_of_each_drug)
        if self.logger is None:
            print("Test {0:.4f}, {1:.4f}, {2:.4f}".format(avg_of_top_aucroc, avg_of_top_prauc, self.epoch))
        else:
            self.logger.debug("Test {0:.4f}, {1:.4f}, {2:.4f}".format(avg_of_top_aucroc, avg_of_top_prauc, self.epoch))
        return avg_of_top_aucroc, avg_of_top_prauc
 


             
        




