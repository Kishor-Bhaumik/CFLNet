import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import timm
from .aspp import build_aspp
from .srm import setup_srm_layer


class CFLNet(nn.Module):
    def __init__(self, cfg, inplanes):
        super(CFLNet, self).__init__()
        self.cfg = cfg
        self.encoder = timm.create_model(self.cfg['model_params']['encoder'], pretrained= True, features_only=True, out_indices=[4])
        self.conv_srm = setup_srm_layer()
        self.encoder_srm = timm.create_model(self.cfg['model_params']['encoder'], pretrained= False, features_only=True, out_indices=[4])
        if self.cfg['global_params']['with_srm'] == True:
            self.aspp = build_aspp(inplanes = inplanes*2, outplanes = self.cfg['model_params']['aspp_outplane'] )
        else:
            self.aspp = build_aspp(inplanes = inplanes, outplanes = self.cfg['model_params']['aspp_outplane'] )
        
            
        self.decoder = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3,
                                              padding=1, stride=1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.Conv2d(512, self.cfg['model_params']['num_class'], kernel_size=1,
                                              stride=1, bias=True))
        self.projection = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 256, kernel_size=1)
            )
    def forward(self, inp):
        x = self.encoder(inp)[0]

        if self.cfg['global_params']['with_srm'] == True:
            x_srm = self.conv_srm(inp)
            x_srm = self.encoder_srm(x_srm)[0]
            x = torch.concat([x, x_srm], dim=1)

        x = self.aspp(x)
        x = F.interpolate(x, scale_factor = 4, mode = 'bilinear', align_corners= True)
        out = self.decoder(x)
        proj = self.projection(x)
        return out, proj