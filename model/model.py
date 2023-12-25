#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2023/4/25 15:55
# @Author : 慕夏
# @File : model.py
# @Project :HyperKG
import decimal

import numpy as np
import torch
from utils.ponicare_utils import *
import torch.nn as nn
from utils.mobius_linear import *


class HyEED(torch.nn.Module):
    def __init__(self, d, dim, config):
        super(HyEED, self).__init__()
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.Eh.weight.data = (1e-3 * torch.randn((len(d.entities), dim), dtype=torch.double, device=config.device))
        # self.Eh.weight.data.copy_(torch.from_numpy(np.array(d.entities_embeddings)))
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rvh.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device=config.device))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), dim)),
                                                  dtype=torch.double, requires_grad=True, device=config.device))
        self.bs = torch.nn.Parameter(
            torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=config.device))
        self.bo = torch.nn.Parameter(
            torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=config.device))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, entity1, relation, entity2):

        entity_1 = self.Eh.weight[entity1]
        entity_2 = self.Eh.weight[entity2]
        relation_Ru = self.Wu[relation]
        relation_rvh = self.rvh.weight[relation]

        entity_1_out = torch.where(torch.norm(entity_1, 2, dim=-1, keepdim=True) >= 1,
                                   entity_1 / (torch.norm(entity_1, 2, dim=1, keepdim=True) - 1e-5),
                                   entity_1)
        entity_2_out = torch.where(torch.norm(entity_2, 2, dim=-1, keepdim=True) >= 1,
                                   entity_2/ (torch.norm(entity_2, 2, dim=1, keepdim=True) - 1e-5),
                                   entity_2)
        relation_rvh = torch.where(torch.norm(relation_rvh, 2, dim=-1, keepdim=True) >= 1,
                                   relation_rvh / (torch.norm(relation_rvh, 2, dim=1, keepdim=True) - 1e-5),
                                   relation_rvh)

        entity_1_log = p_log_map(entity_1_out)
        entity_1_W = entity_1_log * relation_Ru
        entity_1_m = p_exp_map(entity_1_W)
        entity_2_m = p_sum(entity_2_out, relation_rvh)

        entity_1_m = torch.where(torch.norm(entity_1_m, 2, dim=-1, keepdim=True) >= 1,
                                 entity_1_m / (torch.norm(entity_1_m, 2, dim=-1, keepdim=True) - 1e-5), entity_1_m)
        entity_2_m = torch.where(torch.norm(entity_2_m, 2, dim=-1, keepdim=True) >= 1,
                                 entity_2_m / (torch.norm(entity_2_m, 2, dim=-1, keepdim=True) - 1e-5), entity_2_m)

        sqdist = (2. * artanh(torch.clamp(torch.norm(p_sum(-entity_1_m, entity_2_m), 2, dim=-1), 1e-10, 1 - 1e-5))) ** 2


        return -sqdist + self.bs[entity1] + self.bo[entity2]


class NC_Model(torch.nn.Module):
    def __init__(self, d,dim, config):
        super(NC_Model, self).__init__()
        self.c_seed = 1.0
        self.manifold = PoincareBall()
        self.config = config
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.Eh.weight.data = (1e-3 * torch.zeros((len(d.entities), dim), dtype=torch.double, device=config.device))
        # self.Eh.weight.data.copy_(torch.from_numpy(np.array(d.entities_embeddings)))
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rvh.weight.data = (1e-3 * torch.randn((len(d.relations), dim), dtype=torch.double, device=config.device))
        self.Wu = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (len(d.relations), dim)),
                                                  dtype=torch.double, requires_grad=True, device=config.device))
        self.bs = torch.nn.Parameter(
            torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=config.device))
        self.bo = torch.nn.Parameter(
            torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=config.device))
        self.loss = torch.nn.BCEWithLogitsLoss()
        # self.hyperLinear = MobiusLinear(self.manifold, self.config.embed, self.config.num_classes, c=self.c_seed, config=config)


    def forward(self,x):

        entity = self.Eh.weight[x]
        entity = torch.squeeze(entity)
        entity = entity.float()
        out = self.hyperLinear(entity)
        y = self.manifold.logmap0(out,self.c_seed)
        return y