import torch
import torch.nn as nn
import wandb
import numpy as np
from drug_cell_attention import DrugCellAttention
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops,  to_dense_batch
import torch.nn.functional as F
from ltr_loss import point_wise_mse, list_wise_listnet, list_wise_listmle, pair_wise_ranknet, list_wise_rankcosine, \
    list_wise_ndcg, combine_loss, weighted_point_wise_mse, ContrastiveLoss,weighted_point_wise_mse_adjN
from collections import defaultdict
from torch import save
import metric 

class TransProModelBase(nn.Module):

    '''
    provide some basic model functions
    '''
    def __init__(self, device):
        super(TransProModelBase, self).__init__()
        self.device = device
        self.initializer = torch.nn.init.kaiming_uniform_
        self.prediction_ls = []
        self.label_ls = []
        self.loss_ls = []
        self.best_performance = defaultdict(lambda: -float('inf'))

    def config_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for param_group in self.optimizer.param_groups:
            print("============current learning rate is {0!r}".format(param_group['lr']))

    def init_weights(self):
        print('Initialized weights............')
        for name, parameter in self.named_parameters():
            if parameter.dim() == 1:
                nn.init.constant_(parameter, 10**-7)
            else:
                self.initializer(parameter)

    def freeze_modules(self, *frozen_modules):
        # print('frozen the parameters')
        for frozen_module in frozen_modules:
            for param in frozen_module.parameters():
                param.requires_grad = False
    
    def unfreeze_modules(self, *unfrozen_modules):
        # print('Unfreeze the parameters')
        for unfrozen_module in unfrozen_modules:
            for param in unfrozen_module.parameters():
                param.requires_grad = True
    
    def load_weights(self, sub_module, saved_sub_module):
        sub_module.load_state_dict(saved_sub_module.state_dict())

    def loss(self, label, predict, loss_type = 'point_wise_mse'):
        if loss_type == 'point_wise_mse':
            loss = point_wise_mse(label, predict)
        elif loss_type == 'pair_wise_ranknet':
            loss = pair_wise_ranknet(label, predict, self.device)
        elif loss_type == 'list_wise_listnet':
            loss = list_wise_listnet(label, predict)
        elif loss_type == 'list_wise_listmle':
            loss = list_wise_listmle(label, predict, self.device)
        elif loss_type == 'list_wise_rankcosine':
            loss = list_wise_rankcosine(label, predict)
        elif loss_type == 'list_wise_ndcg':
            loss = list_wise_ndcg(label, predict)
        elif loss_type == 'combine':
            loss = combine_loss(label, predict, self.device)
        elif loss_type == 'weighted_point_wise_mse':
            loss = weighted_point_wise_mse(label, predict)
        elif loss_type == 'weighted_point_wise_mse_adjN':
            loss = weighted_point_wise_mse_adjN(label, predict)
        elif loss_type == 'contrastive_loss':
            batch_size = label.size(0)
            contrastive_loss = ContrastiveLoss(batch_size=batch_size).to(self.device)
            loss = contrastive_loss(label, predict)
        else:
            raise ValueError('Unknown loss: %s' % loss_type)
        return loss

    def train_epoch_end(self, epoch, model_persistence_dir = None):
        print('Train loss:')
        print(np.mean(self.loss_ls))
        wandb.log({'Train loss': np.mean(self.loss_ls)}, step = epoch)
        self.__reset_result_ls()
        if epoch % 100 == 0:
            if model_persistence_dir:
                prefix, suffix = model_persistence_dir.rsplit('.', 1)
                model_persistence_dir = prefix + '_{}_'.format(epoch) + suffix
            self.model_persistence(model_persistence_dir)
    
    def model_persistence(self, model_persistence_dir):

            if model_persistence_dir:
                save(self.state_dict(), model_persistence_dir)
    
    def validation_test_epoch_end(self, epoch,  metrics_summary=None, validation_test_flag = 'Validation', model_persistence_dir = None ):
        lb_np, predict_np = np.concatenate(self.label_ls), np.concatenate(self.prediction_ls)
        print('{} loss:'.format(validation_test_flag))
        print(np.mean(self.loss_ls))
        wandb.log({'{} loss'.format(validation_test_flag): np.mean(self.loss_ls)}, step=epoch)
        rmse = metric.rmse(lb_np, predict_np, remove_zero = True)
        print('RMSE: %.4f' % rmse)
        wandb.log({'{} RMSE'.format(validation_test_flag): rmse}, step=epoch)
        pearson, _ = metric.correlation(lb_np, predict_np, 'pearson', remove_zero = True)
        print('Pearson\'s correlation: %.4f' % pearson)
        wandb.log({'{} Pearson'.format(validation_test_flag): pearson}, step = epoch)
        spearman, _ = metric.correlation(lb_np, predict_np, 'spearman', remove_zero = True)
        print('Spearman\'s correlation: %.4f' % spearman)
        wandb.log({'{} Spearman'.format(validation_test_flag): spearman}, step = epoch)
        perturbed_precision = []
        for k in [10, 20]:
            precision_neg, precision_pos = metric.precision_k(lb_np, predict_np, k, remove_zero = True)
            print("Precision@%d Positive: %.4f" % (k, precision_pos))
            print("Precision@%d Negative: %.4f" % (k, precision_neg))
            perturbed_precision.append([precision_pos, precision_neg])
        self.__reset_result_ls()
        if 'Validation' in validation_test_flag:
            if self.best_performance[validation_test_flag] < pearson:
                self.best_performance[validation_test_flag] = pearson
                self.model_persistence(model_persistence_dir)

        if  validation_test_flag == 'Perturbed_Pros_Validation':
            metrics_summary['pearson_list_dev'].append(pearson)
            metrics_summary['spearman_list_dev'].append(spearman)
            metrics_summary['rmse_list_dev'].append(rmse)
         
        if  validation_test_flag =='Perturbed_Pros_Test':
            metrics_summary['pearson_list_test'].append(pearson)
            metrics_summary['spearman_list_test'].append(spearman)
            metrics_summary['rmse_list_test'].append(rmse)

    def __reset_result_ls(self):
        self.loss_ls, self.prediction_ls, self.label_ls = [], [], []
 
class TransProModel(TransProModelBase):

    '''
    transPro model
    '''
    def __init__(self, device, transPro_config,args):

        super(TransProModel, self).__init__(device)
        self.config = transPro_config
        self.cell_encoder = twol_MLP(
                            input_dim= self.config.cell_encoder.cell_dim ,
                            output_dim= self.config.cell_encoder.latent_space_dim,
                            hidden_dim=self.config.cell_encoder.cell_hidden_dim,
                            dop=args.dop )
        self.pros_decoder = twol_MLP(
                            input_dim= self.config.pros_decoder.input_dim,
                            output_dim= self.config.pros_decoder.output_dim,
                            hidden_dim=self.config.pros_decoder.hidden_dim,
                            dop=args.dop )
        self.trans_decoder = twol_MLP(
                            input_dim= self.config.trans_decoder.input_dim,
                            output_dim= self.config.trans_decoder.output_dim,
                            hidden_dim=self.config.trans_decoder.hidden_dim,
                            dop=args.dop )                
        self.drug_network = GNN(
                            num_layer=self.config.drug_network.num_layer,
                            emb_dim=self.config.drug_network.emb_dim,
                            JK=self.config.drug_network.JK,
                            drop_ratio=args.dop,
                            gnn_type=self.config.drug_network.gnn_type)
        self.diff_generator = twol_MLP(
                            input_dim= self.config.diff_generator.input_dim ,
                            output_dim= self.config.diff_generator.output_dim,
                            hidden_dim=self.config.diff_generator.hidden_dim, 
                            dop=args.dop )
        self.drug_cell_attn = DrugCellAttention(self.config.drug_cell_attn.hidden_dim, self.config.drug_cell_attn.hidden_dim, n_layers=2, n_heads=4, pf_dim=512,
                                                dropout=args.dop, device=device)
    
        #self.guassian_noise = GaussianNoise(device=device)
        self.transmitter = twol_MLP(
                            input_dim= self.config.transmitter.input_dim ,
                            output_dim= self.config.transmitter.output_dim,
                            hidden_dim=self.config.transmitter.hidden_dim, 
                            dop=args.dop )
        self.infer_mode = args.infer_mode 
        self.use_transmitter = args.use_transmitter
    def forward(self, drug_feature, cell_feature, job, epoch = 0):

        if job == 'perturbed_trans':
            return self.perturbed_trans_forward(drug_feature, cell_feature, epoch = epoch)
        else: ## 'perturbed_pros'
            return self.perturbed_pros_forward(drug_feature, cell_feature, epoch = epoch)
    def freeze_modules(self, *frozen_modules):
        # print('frozen the parameters')
        for frozen_module in frozen_modules:
            for param in frozen_module.parameters():
                param.requires_grad = False
    
    def unfreeze_modules(self, *unfrozen_modules):
        # print('Unfreeze the parameters')
        for unfrozen_module in unfrozen_modules:
            for param in unfrozen_module.parameters():
                param.requires_grad = True

    def load_pretrained_modules(self, pretrain_model):
        print('Loading in weights')
        self.load_weights(self.cell_encoder, pretrain_model.trans_encoder)
        print('Loaded in cell encoder weights')
        self.load_weights(self.trans_decoder, pretrain_model.trans_decoder)
        print('Loaded in trans decoder weights')
        self.load_weights(self.pros_decoder, pretrain_model.pros_decoder)
        print('Loaded in pros decoder weights')
        if self.use_transmitter==1:
            self.load_weights(self.transmitter, pretrain_model.transmitter)
            print('Loaded in transmitter weights')

    def perturbed_trans_train_step(self, drug_feature, cell_feature, labels, epoch = 0):
        self.optimizer.zero_grad()
        self.train()
       
        predict = self.forward(drug_feature, cell_feature, 'perturbed_trans', epoch)
        loss = self.loss(labels, predict, self.config.perturbed_trans.loss_type)
        loss.backward()
        
        self.optimizer.step()
        self.loss_ls.append(loss.item())

    def perturbed_pros_train_step(self, drug_feature, cell_feature, labels, epoch = 0,freeze_pretrained_modules=0):
        self.optimizer.zero_grad()
        self.train()
        if freeze_pretrained_modules ==1:
                self.freeze_modules(self.cell_encoder, 
                                    self.drug_network,
                                    self.diff_generator)
                                    #self.drug_cell_attn)    

        predict = self.forward(drug_feature, cell_feature, 'perturbed_pros', epoch)
        loss = self.loss(labels, predict, self.config.perturbed_pros.loss_type)
        loss.backward()
       
        self.optimizer.step()
        self.loss_ls.append(loss.item())

    def perturbed_trans_val_test_step(self, drug_feature, cell_feature, labels, epoch = 0):
        self.eval()
        predict = self.forward(drug_feature, cell_feature, 'perturbed_trans', epoch)
        if self.infer_mode ==0:
            loss = self.loss(labels, predict, self.config.perturbed_trans.loss_type)
            self.loss_ls.append(loss.item())    
        self.label_ls.append(labels.cpu().numpy())
        self.prediction_ls.append(predict.detach().cpu().numpy())        

    def perturbed_pros_val_test_step(self, drug_feature, cell_feature, labels, epoch = 0):
        self.eval()
        predict = self.forward(drug_feature, cell_feature, 'perturbed_pros', epoch)
        if self.infer_mode ==0 or self.infer_mode ==2:
            loss = self.loss(labels, predict, self.config.perturbed_pros.loss_type)
            self.loss_ls.append(loss.item())
        self.label_ls.append(labels.cpu().numpy())
        self.prediction_ls.append(predict.detach().cpu().numpy())        

    def perturbed_trans_forward(self, drug_feature, cell_feature, epoch = 0):
        _,cell_embed = self.cell_encoder(cell_feature)
        drug_atom_embed = self.drug_network(drug_feature.x,drug_feature.edge_index,drug_feature.edge_attr)
        drug_atom_embed ,mask = to_dense_batch(drug_atom_embed,drug_feature.batch)
        drug_embed = torch.sum(drug_atom_embed, dim=1)
        _,drug_diff = self.diff_generator(drug_embed)
        drug_cell_embed, _ = self.drug_cell_attn(cell_embed, drug_diff,None, None)
         # choose either cat or addition. if cat, change the model dim 
        # cat 
        drug_cell_embed = torch.cat((drug_cell_embed , drug_diff/10.0),dim=1)
        # addition 
        #drug_cell_embed = drug_cell_embed + drug_diff/10.0
        if self.infer_mode ==0:
            return self.trans_decoder(drug_cell_embed)[1]
        else: 
            return self.trans_decoder(drug_cell_embed)[0]

    def perturbed_pros_forward(self, drug_feature, cell_feature, epoch = 0):
        #add gn on cell encoder 
        #cell_feature = self.guassian_noise(cell_feature)
        _,cell_embed = self.cell_encoder(cell_feature)
        drug_atom_embed = self.drug_network(drug_feature.x,drug_feature.edge_index,drug_feature.edge_attr)
        drug_atom_embed ,mask = to_dense_batch(drug_atom_embed,drug_feature.batch)
        drug_embed = torch.sum(drug_atom_embed, dim=1)
        _,drug_diff = self.diff_generator(drug_embed)
        drug_cell_embed, _ = self.drug_cell_attn(cell_embed, drug_diff,None, None)
        # choose either cat or addition. if cat, change the model dim 
        # cat 
        drug_cell_embed = torch.cat((drug_cell_embed , drug_diff),dim=1)
        # addition 
        #drug_cell_embed = drug_cell_embed + drug_diff
        # w transmitter 
        if self.use_transmitter==1:
            _,drug_cell_embed = self.transmitter(drug_cell_embed)
        if self.infer_mode ==0 or self.infer_mode ==2 :
            return self.pros_decoder(drug_cell_embed)[1]
        else: 
            #return self.pros_decoder(drug_cell_embed)[0]
            return drug_cell_embed

num_atom_type = 120  # including the extra mask tokens
num_degree = 11
num_formal_charge = 11
num_hybrid = 7
num_aromatic = 2
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        
    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GNN(torch.nn.Module):
    """
    
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_degree, emb_dim)
        self.x_embedding3 = torch.nn.Embedding(num_formal_charge, emb_dim)
        self.x_embedding4 = torch.nn.Embedding(num_hybrid, emb_dim)
        self.x_embedding5 = torch.nn.Embedding(num_aromatic, emb_dim)
        self.x_embedding6 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding6.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add"))
            # elif gnn_type == "gcn":
            #     self.gnns.append(GCNConv(emb_dim))
            # elif gnn_type == "gat":
            #     self.gnns.append(GATConv(emb_dim))
            # elif gnn_type == "graphsage":
            #     self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1]) + self.x_embedding3(x[:,2]) + self.x_embedding4(x[:,3]) + self.x_embedding5(x[:,4]) + self.x_embedding6(x[:,5])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation

class twol_MLP(nn.Module):
    def __init__(self,input_dim: int, output_dim:int, hidden_dim: int, 
                dop: float =0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,output_dim)
        self.dop = nn.Dropout(dop)
        self.act = nn.ReLU()
    def forward(self,x): 
        hidden1 = self.fc1(x)
        hidden2 = self.act(hidden1)
        hidden2 = self.dop(hidden2)
        out = self.fc2(hidden2)
        out = self.dop(out)
        return hidden1,out 


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.
    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True, device = 'cpu'):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.std().detach() if self.is_relative_detach else self.sigma * x.std()
            sampled_noise = self.noise.repeat(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x 


