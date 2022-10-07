from ast import arg
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from tqdm import tqdm
from utils.utils import AverageMeter, batch_intersection_union, set_random_seed
from sklearn import metrics
import timm
import yaml
from model.model import CFLNet
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model', action='store', type=str, help='evaluate on test data with pretrained model')
args = parser.parse_args()
SEED= 1221
set_random_seed(SEED)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
with open('config/config.yaml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

with torch.no_grad():
    test_model = timm.create_model(cfg['model_params']['encoder'], pretrained= False, features_only=True, out_indices=[4])
    in_planes = test_model(torch.randn((2,3,128,128)))[0].shape[1]
    del test_model

model = CFLNet(cfg, in_planes).to(device)
from dataloader.loader import generator
model.load_state_dict(torch.load('best_model/' + args.pretrained_model))


gnr = generator(cfg)
validation_generator = gnr.get_val_generator()

casia_imbalance_weight = torch.tensor(cfg['dataset_params']['imbalance_weight']).to(device)
criterion = nn.CrossEntropyLoss(weight = casia_imbalance_weight)

if __name__ == '__main__':
    with torch.no_grad():
        model.eval()
        val_inter = AverageMeter()
        val_union = AverageMeter()
        val_pred = []
        val_tar = []
        auc = []
        kk=1
        f1=[]
        for img, tar in tqdm(validation_generator):
        #for img, tar in tqdm(validation_generator):
            img, tar = img.to(device), tar.to(device)
            pred, _= model(img)
            pred = F.interpolate(pred, img.shape[2:], mode= 'bilinear', align_corners = True)

            intr, uni = batch_intersection_union(pred, tar, num_class = cfg['model_params']['num_class'])
            val_inter.update(intr)
            val_union.update(uni)
            y_score = F.softmax(pred, dim=1)[:,1,:,:]
            for yy_true, yy_pred in zip(tar.cpu().numpy(), y_score.cpu().numpy()) :
            
                this = metrics.roc_auc_score(yy_true.astype(int).ravel(), yy_pred.ravel(), average = None)
                this = metrics.roc_auc_score(yy_true.astype(int).ravel(), yy_pred.ravel(), average = None)
                thisf = metrics.f1_score(yy_true.astype(int).ravel(), (yy_pred > .5).ravel())
                that = metrics.roc_auc_score(yy_true.astype(int).ravel(), (1-yy_pred).ravel(), average = None)
                thatf = metrics.f1_score(yy_true.astype(int).ravel(), (1 - yy_pred > .5).ravel())
                auc.append(max(this,that))
                f1.append(max(thisf, thatf))
                
        


    val_auc = np.mean(auc)
    val_f1 = np.mean(f1)

    logs = { 'Validation AUC': val_auc, 'validation f1': val_f1}
    print(logs)


