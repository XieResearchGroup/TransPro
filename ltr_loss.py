import torch
import torch.nn.functional as F
import loss_utils
import pdb
import torch.nn as nn
from sklearn.metrics import mean_squared_error

def point_wise_mse(label, predict):
    loss = loss_utils.mse(label, predict)
    return loss

def weighted_point_wise_mse(label, predict):
    weighted_loss = torch.sum(
        (torch.where(label==0, 0, 1)) * ((label.float() - predict) ** 2)) / (
            label.size(0) * label.size(1))
    return weighted_loss

def weighted_point_wise_mse_adjN(label, predict):
    weighted_loss = torch.sum(
        (torch.where(label==0, 0, 1)) * ((label.float() - predict) ** 2)) / (
           (label!=0).sum())
    return weighted_loss
def weighted_point_wise_mse_mask(label, predict):
    mask = label !=0
    weighted_loss = mean_squared_error (label[mask],predict[mask])
    return weighted_loss

def mse_plus_homophily(label, predict, hidden_rep, cell_type):

    loss = point_wise_mse(label, predict) + 0.5 * loss_utils.apply_NodeHomophily(hidden_rep, cell_type)
    return loss

def classification_cross_entropy(label, predict):
    shape = predict.size()
    label = label.view(shape[0] * shape[1])
    predict = predict.view(shape[0] * shape[1], shape[2])
    loss = loss_utils.ce(predict, label)
    return loss


def pair_wise_ranknet(label, predict, device):
    """
    From RankNet to LambdaRank to LambdaMART: An Overview
    :param predict: [batch, ranking_size]
    :param label: [batch, ranking_size]
    :return:
    """
    pred_diffs = torch.unsqueeze(predict, dim=2) - torch.unsqueeze(predict, dim=1)  # computing pairwise differences, i.e., Sij or Sxy
    pred_pairwise_cmps = loss_utils.tor_batch_triu(pred_diffs, k=1, device=device) # k should be 1, thus avoids self-comparison
    tmp_label_diffs = torch.unsqueeze(label, dim=2) - torch.unsqueeze(label, dim=1)  # computing pairwise differences, i.e., Sij or Sxy
    std_ones = torch.ones(tmp_label_diffs.size()).to(device).double()
    std_minus_ones = std_ones - 2.0
    label_diffs = torch.where(tmp_label_diffs > 0, std_ones, tmp_label_diffs)
    label_diffs = torch.where(label_diffs < 0, std_minus_ones, label_diffs)
    label_pairwise_cmps = loss_utils.tor_batch_triu(label_diffs, k=1, device=device)  # k should be 1, thus avoids self-comparison
    loss_1st_part = (1.0 - label_pairwise_cmps) * pred_pairwise_cmps * 0.5   # cf. the equation in page-3
    loss_2nd_part = torch.log(torch.exp(-pred_pairwise_cmps) + 1.0)    # cf. the equation in page-3
    loss = torch.sum(loss_1st_part + loss_2nd_part)
    return loss


def list_wise_listnet(label, predict):
    label = F.softmax(label, dim=1)
    predict = F.softmax(predict, dim=1)
    loss = -(label * torch.log(predict)).sum(dim=1).mean()
    return loss


def list_wise_listmle(label, predict, device):
    shape = label.size()
    index = torch.argsort(label, descending=True)
    tmp = torch.zeros(shape[0] * shape[1], dtype=torch.int64).to(device)
    for i in range(0, shape[0] * shape[1], shape[1]):
        tmp[i:(i + shape[1])] += i
    index = index.view(shape[0] * shape[1])
    index += tmp
    predict = predict.view(shape[0] * shape[1])
    predict = predict[index]
    predict = predict.view(shape[0], shape[1])
    predict_logcumsum = loss_utils.apply_LogCumsumExp(predict)
    loss = (predict_logcumsum - predict).sum(dim=1).mean()
    return loss


def list_wise_rankcosine(label, predict):
    loss = torch.sum((1.0 - loss_utils.cos(predict, label)) / 0.5)
    return loss

def list_wise_ndcg(label, predict):
    approx_nDCG = loss_utils.apply_ApproxNDCG_OP(predict, label)
    loss = -torch.mean(approx_nDCG)
    return loss


def combine_loss(label, predict, device):
    mse_loss = point_wise_mse(label, predict)
    listmle_loss = list_wise_rankcosine(label, predict)
    loss = mse_loss + listmle_loss
    return loss


def pearson(x, y):
    mean_x = torch.mean(x, dim=-1, keepdim=True)
    mean_y = torch.mean(y, dim=-1, keepdim=True)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = torch.sum(xm * ym, dim=-1)
    r_den = torch.norm(xm, 2, dim=-1) * torch.norm(ym, 2, dim=-1)
    r_val = r_num / r_den
    return - torch.sum(r_val)

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size = 32, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
