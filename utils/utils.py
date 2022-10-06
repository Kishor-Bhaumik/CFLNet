import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import sys
import random

def get_dataset_mean(training_generator):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in training_generator:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

def batch_intersection_union(predict, target, num_class):
    _, predict = torch.max(predict, 1)
    predict = predict + 1 #no intersection indexes will be 0. Overlaps with background, thus, as a solution, is why all indexes are incrased by 1
    target = target + 1

    predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count
            
def write_logger(filename_log, cfg, **kwargs):
    if not os.path.isdir('results'):
        os.mkdir('results')
    f = open('results/'+filename_log, "a")
    if kwargs['epoch'] == 0:
        f.write("Training CONFIGS are: "+ 
        "SRM="+str(cfg['global_params']['with_srm'])+ " "+ "Contrastive="+ str(cfg['global_params']['with_con']) + " " 
        +"Encoder Name: "+cfg['model_params']['encoder'] + "\n" )
        f.write("\n")
    

    for key, value in kwargs.items():
        f.write(str(key) +": " +str(value)+ "\n")
    f.write("\n")
    f.close()

def set_random_seed(seed, deterministic=False):
    """Set random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
def calfeaturevectors(feat, mask):
    #feat and mask both should be BXCXHXW
    out = torch.mul(feat.unsqueeze(dim=1), mask.unsqueeze(dim=2)).float()
    sum_feat = torch.sum(out, dim=(3,4))
    mask_sum = torch.sum(mask, dim=(2,3))
    mean_feat = sum_feat/((mask_sum+1e-16).unsqueeze(dim=2))
    return mean_feat

def one_hot(label, n_class, device = None):
    #label should be BXHXW
    B,H,W = label.shape
    encoded = torch.zeros(size=(B,n_class,H,W)) 
    if device:
        encoded = encoded.to(device)
    encoded = encoded.scatter_(1, label.unsqueeze(1), 1)
    return encoded

def coral(source, target):

    d = source.size(1)  # dim vector

    source_c = compute_covariance(source)
    target_c = compute_covariance(target)

    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

    loss = loss / (4 * d * d)
    return loss


def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)

    return c

class Get_Patch():
    def __init__(self,cfg):
        self.cfg = cfg
    def check_edge(self, H, W, h, w, p_len):
        return  (H - h > p_len)  and (W - w > p_len)

    def check_all_pristine(self, mask, h, w, p_len):
        return (mask[h:h+p_len, w:w+p_len] == 0).all().item()

    def check_all_tampered(self, mask, h, w, p_len):
        return (mask[h:h+p_len, w:w+p_len] == 1).all().item()

    def get_pristine_patch(self, mask, p_len, threshold):
        H, W = mask.shape
        indices = (mask == 0).nonzero(as_tuple=False)
        t_count = 0
        while True:
            rand_ind = torch.randint(0, len(indices),  (1,)).item()
            h, w = indices[rand_ind][0].item(), indices[rand_ind][1].item()
            t_count += 1
            if self.check_edge(H, W, h, w, p_len) and self.check_all_pristine(mask, h, w, p_len):
                break
            if t_count >= threshold:
                return torch.zeros_like(mask)
            
        p = torch.zeros_like(mask)
        p[h:h+p_len, w:w+p_len] = torch.ones(p_len, p_len)
        return p

    def get_spliced_patch(self, mask, p_len, threshold):
        H, W = mask.shape
        indices = (mask == 1).nonzero(as_tuple=False)
        t_count = 0
        while True:
            rand_ind = torch.randint(0, len(indices),  (1,)).item()
            h, w = indices[rand_ind][0].item(), indices[rand_ind][1].item()
            t_count += 1
            if self.cfg['dataset_params']['take_tampered_boundary'] == True:
                if self.check_edge(H, W, h, w, p_len) and not self.check_all_tampered(mask, h, w, p_len):
                    break
            else:
                if self.check_edge(H, W, h, w, p_len) and self.check_all_tampered(mask, h, w, p_len):
                    break
            if t_count >= threshold:
                return torch.zeros_like(mask)
            
        p = torch.zeros_like(mask)
        p[h:h+p_len, w:w+p_len] = torch.ones(p_len, p_len)
        return p
    
def patch_contrast(pos, neg, device, temperature):
    B = pos.shape[0]
    cat = torch.concat((pos, neg), dim=1)
    
    # import pdb; pdb.set_trace()
    logits = torch.matmul(cat, torch.transpose(cat, 1,2))
    logits = torch.div(logits, temperature)

    identity = torch.eye(logits.shape[-1]).to(device).detach()
    neg_identity = torch.add(torch.negative(identity),1).detach()

    # logits = torch.mul(logits, neg_identity)
    
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()

    logits = torch.exp(logits)

    # logits = torch.mul(logits, neg_identity)
   

    pos_labels = torch.ones((pos.shape[1])).to(device)
    neg_labels = torch.zeros((neg.shape[1])).to(device)
    labels = torch.cat((pos_labels, neg_labels)).contiguous().view(-1,1)
    mask = torch.eq(labels, labels.T).float().detach()
    mask_neg = torch.add(torch.negative(mask),1).detach()

    neg_sum = torch.sum(torch.mul(logits, mask_neg), dim=-1).contiguous().view(B,-1,1)
    denominator_sum = torch.add(logits, neg_sum)
    division = torch.div(logits, denominator_sum+ 1e-18)
    
    loss_matrix = -torch.log(division+1e-18)
    loss_matrix = torch.mul(loss_matrix , mask)
    
    loss_matrix = torch.mul(loss_matrix, neg_identity)
    loss = torch.sum(loss_matrix, dim=-1)
    
    loss = torch.div(loss, mask.sum(dim=-1) -1 + 1e-18) #cardinality
    return loss

class memory_bank():
    def __init__(self, length, n_class, featlength, device):
        self.memory = torch.randn(length, n_class, featlength)
        if device:
            self.memory = self.memory.to(device)

    def enqueue(self,feat):
        batch_size = feat.shape[0]
        self.memory = torch.cat((feat, self.memory), dim=0)[:-batch_size]
    

def contrast_loss (feat, mem, device, temperature = 0.6):
    mem_cardinality = mem.shape[0]

    feat_labels = torch.arange(0,feat.shape[1],1).contiguous().view(-1,1)
    mem_labels = torch.arange(0,mem.shape[1],1).contiguous().view(-1,1)

    feat_labels = feat_labels.repeat((1,feat.shape[0])).contiguous().view(-1,1)
    mem_labels = mem_labels.repeat((1,mem.shape[0])).contiguous().view(-1,1)

    feat = torch.cat(torch.unbind(feat, dim=1), dim=0)
    mem = torch.cat(torch.unbind(mem, dim=1), dim=0)

    # feat_mask = torch.eq(feat_labels, feat_labels.T).float().to(device)
    mem_mask = torch.eq(feat_labels, mem_labels.T).float().to(device)


    # feat_mask_neg = torch.add(torch.negative(feat_mask),1).to(device)
    mem_mask_neg = torch.add(torch.negative(mem_mask),1).to(device)

    feat_logits =  torch.div(torch.matmul(feat, feat.T),temperature)
    mem_logits = torch.div(torch.matmul(feat, mem.T),temperature)

    # for stability
    feat_logits_max, _ = torch.max(feat_logits, dim=1, keepdim=True)
    feat_logits = feat_logits - feat_logits_max.detach()
    mem_logits_max, _ = torch.max(mem_logits, dim=1, keepdim=True)
    mem_logits = mem_logits - mem_logits_max.detach()

    feat_logits = torch.exp(feat_logits)
    mem_logits = torch.exp(mem_logits)

    neg_sum = torch.sum(torch.mul(mem_logits, mem_mask_neg), dim=-1).contiguous().view(-1,1)
    denominator = torch.add(mem_logits, neg_sum)
    division = torch.div(mem_logits, denominator+ 1e-18)
    loss_matrix = -torch.log(division+1e-18)
    loss_matrix = torch.mul(loss_matrix , mem_mask)
    loss = torch.sum(loss_matrix, dim=-1)
    loss = torch.div(loss, mem_cardinality)
    return loss

def square_patch_contrast_loss(feat, mask, device, temperature = 0.6):
    #feat shape should be (Batch, Total_Patch_number, Feature_dimension)
    #mask should be (Batch, H, W)

    mem_mask = torch.eq(mask, mask.transpose(1,2)).float()
    mem_mask_neg = torch.add(torch.negative(mem_mask),1)


    feat_logits =  torch.div(torch.matmul(feat, feat.transpose(1,2)),temperature)
    identity = torch.eye(feat_logits.shape[-1]).to(device)
    neg_identity = torch.add(torch.negative(identity),1).detach()

    feat_logits = torch.mul(feat_logits, neg_identity)

    feat_logits_max, _ = torch.max(feat_logits, dim=1, keepdim=True)
    feat_logits = feat_logits - feat_logits_max.detach()

    feat_logits = torch.exp(feat_logits)

    neg_sum = torch.sum(torch.mul(feat_logits, mem_mask_neg), dim=-1)
    denominator = torch.add(feat_logits, neg_sum.unsqueeze(dim=-1))
    division = torch.div(feat_logits, denominator+ 1e-18)
        
    loss_matrix = -torch.log(division+1e-18)
    loss_matrix = torch.mul(loss_matrix , mem_mask)
    loss_matrix = torch.mul(loss_matrix, neg_identity)
    loss = torch.sum(loss_matrix, dim=-1)

    loss = torch.div(loss, mem_mask.sum(dim=-1) -1 + 1e-18)

    return loss
