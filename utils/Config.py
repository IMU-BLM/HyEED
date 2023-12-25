#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2023/5/10 09:35
# @Author : 慕夏
# @File : Config.py
# @Project :TextHyperKG


import os
import torch
import numpy as np

class Config(object):
    def __init__(self, datasetdir, outputdir, embedding):
        self.model_name = 'poincare'
        self.dataset = 'FB15k-237'
        self.data_path = datasetdir
        self.train_path = os.path.join(datasetdir, 'FB15k-237/ind-train.txt')
        self.dev_path = os.path.join(datasetdir, 'FB15k-237/ind-valid.txt')
        self.test_path = os.path.join(datasetdir, 'FB15k-237/ind-test.txt')
        self.description_path = os.path.join(datasetdir,'entity_word/entityWords.txt')
        self.vocab_path = os.path.join(datasetdir, 'entity_word/word2id.txt')
        self.save_path = os.path.join(outputdir, self.dataset + self.model_name + '.pt')
        self.log_path = os.path.join(outputdir, self.model_name + '.log')
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        self.embedding_pretrained = torch.tensor(
            np.load(os.path.join(datasetdir, embedding))["embeddings"].astype(
                'float32')) if embedding != 'random' else None

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.dropout = 0.5
        self.require_improvement = 2500
        self.n_vocab = 0
        self.nneg = 50
        self.num_epochs = 30
        self.wordNgrams = 2
        self.batch_size = 32
        self.max_length = 100
        self.learning_rate = 1e-2
        self.embed = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 100
        self.bucket = 20000
        self.lr_decay_rate = 0.96
        self.num_classes = 50
