#!/usr/bin/env python
# -*- coding: utf-8 -*-
# time: 2023/5/5 15:48
# @Project : HyperKG
# file: data_utils2.py
# author: 慕夏

import os
import torch
import random
from datetime import timedelta
import torch.utils.data as Data
from tqdm import tqdm
import numpy as np
from utils.ponicare_utils import *
# from utils.mobius_linear import *
from collections import defaultdict
from collections import OrderedDict
UNK, PAD = '<UNK>','<PAD>'
MAX_VOCAB_SIZE = 5000000

class GroupDataset:
    def __init__(self, data_dir,dataset,config):
        self.config = config
        self.c_seed = 1.0
        self.manifold = PoincareBall()
        self.dataset = dataset
        self.train_data = self.load_data(data_dir,self.dataset, self.dataset +"/train.tsv")
        print("**************************************")
        print(len(self.train_data))
        self.valid_data = self.load_data(data_dir,self.dataset, self.dataset +"/dev.tsv")
        print(len(self.valid_data))
        self.test_data = self.load_data(data_dir,self.dataset, self.dataset +"/test.tsv")
        print(len(self.test_data))
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(data_dir,self.dataset +"/entities2.txt")
        self.entity_idx = self.get_entities_idx(data_dir,self.dataset +"/entities2.txt")
        self.relations = self.get_relations(data_dir,self.dataset +"/relations.txt")
        self.vocab_vector = self.get_vocab_vector(data_dir, "poincare_glove_50x2D_cosh-dist-sq.txt")
        self.entity_words = self.get_entity_words(data_dir,self.dataset +"/new_entityWords2.txt")
        self.er_vocab = self.get_er_vocab()
        self.entity_tokens_embedding = self.entity_tokens_embedding()
        self.entity_embeddings, self.entities_embeddings = self.tokens_embedding()
    def get_drop_entity(self,file_path, file_dtype):
        drop_entity = set()
        file_name = os.path.join(file_path, file_dtype)
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                drop_entity.add(line)
        return drop_entity
    def get_er_vocab(self):
        er_vocab = defaultdict(list)
        for item in self.data:
            er_vocab[(self.entity_idx.get(item[0]),self.relations.get(item[2]))].append(self.entity_idx.get(item[1]))
        return er_vocab


    def tokens_embedding(self):
        entity_embeddings = OrderedDict()
        entities_embeddings = []
        print("Compute the entity description initial vector: einstein_midpoint................................")
        for item in tqdm(self.entity_tokens_embedding):
            tokens_embedding = self.entity_tokens_embedding.get(item)
            tokens_embedding = torch.tensor(tokens_embedding)
            tokens_embedding = tokens_embedding.unsqueeze(0)
            entity_out = self.manifold.einstein_midpoint(tokens_embedding, c=self.c_seed)
            entity_out = entity_out.squeeze()
            entity_out = entity_out.tolist()
            entities_embeddings.append(entity_out)
            entity_embeddings[item] = entity_out
        return entity_embeddings,entities_embeddings

    def entity_tokens_embedding(self):
        entity_tokens_embdding = OrderedDict()
        print("Gets the entity description vector")
        for entity in self.entity_idx:
            tokens = self.entity_words.get(entity)
            tokens_len = len(tokens)
            if tokens_len >= self.config.max_length:
                tokens = tokens[:self.config.max_length]
            else:
                # tokens = tokens + ["PAD"] * (self.config.max_length - tokens_len)
                less_len = self.config.max_length // tokens_len + 1
                tokens = tokens * less_len
                tokens = tokens[:self.config.max_length]
            tokens_embedding = []
            for word in tokens:
                if word != "PAD":
                    word_embedding = self.vocab_vector.get(word)
                else:
                    word_embedding = [0.0] * 100
                tokens_embedding.append(word_embedding)
            entity_tokens_embdding[entity] = tokens_embedding
        return entity_tokens_embdding
    def get_vocab_vector(self,data_path, file_type):
        file_name = os.path.join(data_path,file_type)
        vocab_vector = OrderedDict()
        with open(file_name, "r", encoding="utf-8") as f:
            for i in f:
                line = i.strip().split(' ')
                word =line[0]
                vector = line[1:]
                word_vector = []
                for j in vector:
                    word_vector.append(float(j))
                vocab_vector[word] = word_vector[:100]
        return vocab_vector

    def load_data(self, file_path,dataset, file_dtype):

        file_path = os.path.join(file_path, file_dtype)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                line = line.strip().split('\t')
                entity1 = line[0]
                entity2 = line[2]
                relation = line[1]
                data.append([entity1, entity2, relation])
        if file_dtype == dataset+'/train.tsv':
            print(123)
            data = data + [[i[1], i[0], i[2] + "/reverse"] for i in data]

        return data

    def get_entities(self,file_path, file_dtype):
        file_path = os.path.join(file_path, file_dtype)
        entities = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                entity = line
                entities.append(entity)
        entities_idx = []
        for index, item in enumerate(entities):
            entities_idx.append((item, index))
        return entities_idx

    def get_entities_idx(self,file_path, file_dtype):
        file_path = os.path.join(file_path, file_dtype)
        entity_idx = OrderedDict()
        entities = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                entity = line
                entities.append(entity)

        for index, line in enumerate(entities):
            entity_idx[line] = index
        # entity_idx["UNK"] = len(entity_idx)
        return entity_idx

    def get_relations(self,file_path, file_dtype):
        file_path = os.path.join(file_path, file_dtype)
        relations_id = OrderedDict()
        relations = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                line = line.strip('\n')
                relations.append(line)
                relations_id[line] = index
        relations_len = len(relations_id)
        for index,relation in enumerate(relations):
            item = relation + "/reverse"
            relations_id[item] = index + relations_len
        return relations_id


    def get_entity_words(self,file_path,file_dtype):
        file_path = os.path.join(file_path, file_dtype)
        entity_words = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = line.split('\t')
                if int(line[1]) == 0:
                    continue
                tokens = (line[2]).strip().split(' ')
                entity_words[line[0]] = tokens

        return entity_words

class NodeDataset:
    def __init__(self, data_dir,config):
        self.config = config
        self.datasets = "WN18RR"
        self.train_data = self.load_data(data_dir, self.datasets+"/train-ents-class.txt")
        self.valid_data = self.load_data(data_dir, self.datasets+"/dev-ents-class.txt")
        self.test_data = self.load_data(data_dir, self.datasets+"/test-ents-class.txt")
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(data_dir, self.datasets+"/entities2.txt")
        self.entity_idx = self.get_entities_idx(data_dir, self.datasets+"/entities2.txt")
        self.relations = self.get_relations(data_dir, self.datasets+"/relations.txt")
        self.entity_num_class = self.get_entity_class(data_dir, self.datasets+"/entity_class.txt")

    def get_entity_class(self,file_path, file_dtype):
        entity_class = OrderedDict()
        file_name = os.path.join(file_path, file_dtype)
        with open(file_name, 'r', encoding='utf-8') as f:
            for index,line in enumerate(f):
                line = line.strip('\n')
                entity_class[line] = index

        return entity_class



    def get_drop_entity(self, file_path, file_dtype):
            drop_entity = set()
            file_name = os.path.join(file_path, file_dtype)
            with open(file_name, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip('\n')
                    drop_entity.add(line)
            return drop_entity

    def load_data(self, file_path, file_dtype):
            file_path = os.path.join(file_path, file_dtype)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = []
                for line in f:
                    line = line.strip().split('\t')
                    entity1 = line[0]
                    relation = line[1]
                    data.append([entity1, relation])
            return data

    def get_entities(self, file_path, file_dtype):
            file_path = os.path.join(file_path, file_dtype)
            entities = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip().split('\t')
                    entity = line[0]
                    entities.append(entity)
            entities_idx = []
            for index, item in enumerate(entities):
                entities_idx.append((item, index))
            return entities_idx

    def get_entities_idx(self, file_path, file_dtype):
            file_path = os.path.join(file_path, file_dtype)
            entity_idx = OrderedDict()
            entities = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip().split('\t')
                    entity = line[0]
                    entities.append(entity)

            for index, line in enumerate(entities):
                entity_idx[line] = index
            # entity_idx["UNK"] = len(entity_idx)
            return entity_idx
    def get_relations(self,file_path, file_dtype):
        file_path = os.path.join(file_path, file_dtype)
        relations_id = OrderedDict()
        relations = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                line = line.strip('\n')
                relations.append(line)
                relations_id[line] = index
        relations_len = len(relations_id)
        for index, relation in enumerate(relations):
            item = relation + "/reverse"
            relations_id[item] = index + relations_len
        return relations_id










