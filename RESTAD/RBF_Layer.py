#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn


class RBFLayer(nn.Module):
    def __init__(self, centers_dim):
        super(RBFLayer, self).__init__()
        self.centers = nn.Parameter(torch.Tensor(*centers_dim))
        self.log_gamma = nn.Parameter(torch.Tensor([1.0]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centers, mean=0, std=1)
        nn.init.normal_(self.log_gamma, mean=0, std=1)

    def forward(self, x):
        x = x.unsqueeze(1) - self.centers.unsqueeze(0).unsqueeze(2)
        x = x ** 2
        x = torch.sum(x, dim=-1)
        output = torch.exp(-0.5 * torch.exp(self.log_gamma) * x )  

        final_output = output.permute(0, 2, 1)
        return final_output
    


    
    