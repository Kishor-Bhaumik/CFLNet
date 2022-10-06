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
