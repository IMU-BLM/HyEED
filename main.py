#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2023/4/25 15:55
# @Author : 慕夏
# @File : main.py
# @Project :HyperKG

import argparse
import random
import time
import os

import numpy
import torch
import numpy as np
from utils.Config import Config
# from utils.datautils1 import *
from utils.data_utils2 import *
from model.model import *
from optimizer.optimizer import *
from collections import defaultdict
from utils.ponicare_utils import *
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score,balanced_accuracy_score
import logging
from keras.callbacks import Callback
from sklearn import datasets


parser = argparse.ArgumentParser(description='Enhanced Text KG Embedding')
parser.add_argument("--dataset", type=str, default="WN18RR", nargs="?",
                    help="Which dataser to use: FB15k-237 , WN18RR")
parser.add_argument('--model', type=str, default="poincare", nargs="?",
                    help="Which model to use: ponicare or euclidean.")
parser.add_argument('--use_word_segment', default=True, type=bool,
                    help='True for word, False for char')
#/workspace/dataset/private/Hyper_Text_KG/
parser.add_argument('--datasetdir', default='data/', type=str,
                    help='dataset dir')
#/workspace/outputs/Hyper_Text_KG/
parser.add_argument('--outputdir', default='output/model/', type=str,
                    help='output dir')
parser.add_argument('--embedding', default='random', type=str,
                    help='using random init word embedding or using pretrained')
parser.add_argument("--num_iterations", type=int, default=800, nargs="?",
                    help="Number of iterations.")
parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                    help="Batch size")
parser.add_argument('--dropout', default=0.0, type=float,
                    help='dropout')
parser.add_argument("--nneg", type=int, default=50, nargs="?",
                    help="Number of negative samples.")
parser.add_argument("--lr", type=float, default=50, nargs="?",
                    help="Learning rate.")
parser.add_argument("--dim", type=int, default=100, nargs="?",
                    help="Embedding dimensionality.")
parser.add_argument("--bucket", type=int, default=1500000,
                    help='total ngram bucket size')
parser.add_argument('--wordNgrams', type=int, default=2,
                    help='use max n-grams, eg: 2 or 3 ...')
parser.add_argument('--max_length', default=100, type=int,
                    help='max_length')
parser.add_argument('--eval_per_batchs', default=100, type=int,
                    help='eval_per_batchs')
parser.add_argument('--min_freq', default=1, type=int,
                    help='min word frequents of construct vocab')
parser.add_argument('--lr_decay_rate', default=0.96, type=float,
                    help='lr_decay_rate')  # 0.96,0.87
args = parser.parse_args()


class Experiment:
    def __init__(self, learning_rate=50,
                 dim=40, nneg=50, model='poincare', num_iterations=500, batch_size=128, cuda=False):
        self.model = model
        self.learning_rate = learning_rate
        self.dim = dim
        self.nneg = nneg
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.cuda = cuda
        self.entities = [item[0] for item in d.entities]
        # self.d = d

    def get_negsamples(self):
        entities = []
        for item in d.entities:
            entities.append(item[0])
        return entities

    def get_data_idx(self, data):
        data_idx = []
        for i in data:
            data_idx.append((d.entity_idx.get(i[0]), d.relations.get(i[2]), d.entity_idx.get(i[1])))
        return data_idx

    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab
    def get_dataset_id(self, data):
        data_idx = []
        for item in data:
            entity_1_idx = d.entity_idx.get(item[0])
            entity_2_idx = d.entity_idx.get(item[1])
            relation_idx = d.relations.get(item[2])
            data_idx.append((entity_1_idx,relation_idx,entity_2_idx))
        return data_idx

    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        dev_data_idx = self.get_data_idx(data)

        sr_vocab = self.get_er_vocab(self.get_data_idx(d.data))

        print("Number of valid data points: %d" % len(dev_data_idx))
        print("******************")
        print("Evaluating Step: ......")
        print("")
        random.shuffle(dev_data_idx)
        for i in tqdm(range(0,len(dev_data_idx))):

            data_point = dev_data_idx[i]
            e1_idx = torch.tensor(data_point[0], device=config.device).type(torch.long)
            r_idx = torch.tensor(data_point[1], device=config.device).type(torch.long)
            e2_idx = torch.tensor(data_point[2], device=config.device).type(torch.long)
            predictions_s = model.forward(e1_idx.repeat(len(d.entities)),
                                          r_idx.repeat(len(d.entities)), range(len(d.entities)))

            filt = sr_vocab[(data_point[0], data_point[1])]
            target_value = predictions_s[e2_idx].item()
            predictions_s[filt] = -np.Inf
            predictions_s[e1_idx] = -np.Inf
            predictions_s[e2_idx] = target_value

            sort_values, sort_idx = torch.sort(predictions_s, descending=True)

            sort_idx = sort_idx.cpu().numpy()
            rank = np.where(sort_idx == e2_idx.item())[0][0]
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

        self.information1 = 'Hits @10: {0}'.format(np.mean(hits[9]))
        self.information2 = 'Hits @3: {0}'.format(np.mean(hits[2]))
        self.information3 = 'Hits @1: {0}'.format(np.mean(hits[0]))
        self.information4 = 'MRR: {0}'.format(np.mean(1. / np.array(ranks)))
        logging.info(self.information1)
        logging.info(self.information2)
        logging.info(self.information3)
        logging.info(self.information4)

    def train(self):

        print("Training the %s model ................" % self.model)

        train_dataset = d.train_data
        less_data = self.batch_size - (len(train_dataset) % self.batch_size)
        train_dataset = train_dataset + train_dataset[:less_data]
        print("Number of training data points: %d" % len(train_dataset))
        train_dataset  = self.get_dataset_id(train_dataset)
        if self.model == 'poincare':
            model = HyEED(d, self.dim, config)
            #model.load_state_dict(torch.load(config.save_path))

        param_names = [name for name, param in model.named_parameters()]

        opt = RiemannianSGD(model.parameters(), lr=self.learning_rate, param_names=param_names)
        if self.cuda:
            model.to(config.device)

        print("Starting training..................")
        for epoch in range(1, self.num_iterations + 1):
            start_time = time.time()
            model.train()
            temp = 0
            losses = []
            print("***********************************")
            print("Training Step: ......")
            print("")
            random.shuffle(train_dataset)
            for j in tqdm(range(0, len(train_dataset), self.batch_size)):
                temp += 1
                data_batch = np.array(train_dataset[j: j + self.batch_size])
                filt = d.er_vocab[(data_batch[:,0][0],data_batch[:,1][0])]
                entities = list(range(len(d.entity_idx))).copy()
                for item in filt:
                    entities.remove(item)
                negsamples = np.random.choice(entities,size=(data_batch.shape[0], config.nneg))
                # negsamples = similiar_entity[j: j + self.batch_size,:50]
                e1_idx = torch.tensor(np.tile(np.array([data_batch[:, 0]]).T, (1, negsamples.shape[1] + 1)),device=config.device).type(torch.long)
                r_idx = torch.tensor(np.tile(np.array([data_batch[:, 1]]).T, (1, negsamples.shape[1] + 1)),device=config.device).type(torch.long)
                e2_idx = torch.tensor(np.concatenate((np.array([data_batch[:, 2]]).T, negsamples), axis=1),device=config.device).type(torch.long)

                targets = np.zeros(e1_idx.shape)
                targets[:, 0] = 1
                targets = torch.DoubleTensor(targets).to(config.device)

                opt.zero_grad()
                predictions = model.forward(e1_idx, r_idx, e2_idx)
                loss = model.loss(predictions, targets)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            print("**************************")
            print("第{}轮epoch：".format(epoch))
            print("第{}轮运行时间：{}".format(epoch, time.time() - start_time))
            print("第{}轮平均损失: {}".format(epoch, np.mean(losses)))
            logging.info("第{}轮平均损失: {}".format(epoch, np.mean(losses)))
            if np.mean(losses) == None:
                break
# ********************************************************************************************
            with torch.no_grad():
                if epoch % 100 == 0 or epoch == self.num_iterations:
                    model.eval()
                    print("")
                    print("***************************************************************************")
                    print("Valid Step:")
                    logging.info("Valid Step:.................................")
                    self.evaluate(model, d.valid_data)
                    if epoch % 100 == 0 or epoch == self.num_iterations:
                        print("Test Step:")
                        logging.info("Test Step:.................................")
                        self.evaluate(model, d.test_data)
        try:
            torch.save(model.state_dict(), config.save_path)
            print("Model saved successfully!****************")
            logging.info("Model saved successfully!****************")
        except:
            print("Model saving failure!***********")
            logging.info("Model saving failure!***********")
        try:
            information = []
            information.append(self.information1)
            information.append(self.information2)
            information.append(self.information3)
            information.append(self.information4)
            file_path = os.path.join(args.outputdir,"WN18RR_info.txt")
            with open(file_path,'w',encoding='utf-8') as f:
                for item in information:
                    f.write(item)
                    f.write('\n')
            logging.info("The final verification result is saved successfully!************")
        except:
            logging.info("The final verification result failed to be saved!************")


def node_classification(Node_d,model_path,dim,config):

    nc_model = NC_Model(Node_d,dim,config)
    nc_model.load_state_dict(torch.load(model_path))
    entity_embed = nc_model.state_dict()['Eh.weight']
    entity_embed = p_log_map(entity_embed)

    split_data = dict()
    # train_dataset
    entity_idx = []
    labels = []
    for item in Node_d.train_data:
        entity_idx.append(Node_d.entity_idx.get(item[0]))
        labels.append(Node_d.entity_num_class.get(item[1]))

    x = (entity_embed[entity_idx]).cpu().numpy()
    y = np.array(labels)
    split_data['train'] = (x, y)
    # valid_dataset
    entity_idx = []
    labels = []
    for item in Node_d.valid_data:
        entity_idx.append(Node_d.entity_idx.get(item[0]))
        labels.append(Node_d.entity_num_class.get(item[1]))
    x = (entity_embed[entity_idx]).cpu().numpy()
    y = np.array(labels)
    split_data['dev'] = (x, y)

    # Test_dataset
    entity_idx = []
    labels = []
    for item in Node_d.test_data:
        entity_idx.append(Node_d.entity_idx.get(item[0]))
        labels.append(Node_d.entity_num_class.get(item[1]))
    x = (entity_embed[entity_idx]).cpu().numpy()
    y = np.array(labels)
    split_data['test'] = (x, y)

    x_train, y_train = split_data['train']
    x_dev,y_dev = split_data['dev']
    x_test, y_test = split_data['test']
    best_dev_metric = 0.0
    best_c = 0

    for k in range(-6, 5):
        c = 10 ** -k
        model = LogisticRegression(C = c,multi_class='multinomial',max_iter=1000,solver="sag")
        model.fit(x_train,y_train)

        dev_preds = model.predict(x_dev)
        dev_acc = accuracy_score(y_dev,dev_preds)
        bal_acc = balanced_accuracy_score(y_dev,dev_preds)

        test_preds = model.predict(x_test)
        # test_acc = accuracy_score(y_test, test_preds)
        # test_bal = balanced_accuracy_score(y_test, test_preds)
        print("C={}".format(c))
        print("dev_acc:{}".format(dev_acc))
        print("bal_acc:{}".format(bal_acc))

        if dev_acc > best_dev_metric:
            best_dev_metric =dev_acc
            best_c =c

    nc_model = LogisticRegression(C=best_c,multi_class='multinomial',max_iter=1000,solver="sag")
    x_train_all = np.concatenate((x_dev,x_train))
    y_train_all = np.concatenate((y_dev,y_train))
    nc_model.fit(x_train_all,y_train_all)

    test_preds = nc_model.predict(x_test)
    test_acc = accuracy_score(y_test,test_preds)
    test_bal = balanced_accuracy_score(y_test,test_preds)
    print("test_acc:{}".format(test_acc))
    print("test_acc:{}".format(test_bal))

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICE'] = '0'
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)
    if torch.cuda.is_available():
        logging.info("cuda Yes!")
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    dataset_dir = args.datasetdir
    dataset = args.dataset
    print(dataset)
    outputdir = args.outputdir

    embedding = args.embedding

    config = Config(dataset_dir, outputdir, embedding)

    # reset config
    config.model_name = args.model
    config.save_path = os.path.join(outputdir, args.dataset + args.model + '.ckpt')
    config.log_path = os.path.join(outputdir, args.model + '.log')
    config.dropout = float(args.dropout)
    # config.require_improvement = int(args.require_improvement)
    # config.num_epochs = int(args.num_epochs)
    config.batch_size = int(args.batch_size)
    config.max_length = int(args.max_length)
    config.learning_rate = float(args.lr)
    config.embed = int(args.dim)
    config.nneg = int(args.nneg)
    config.bucket = int(args.bucket)
    config.wordNgrams = int(args.wordNgrams)
    config.lr_decay_rate = float(args.lr_decay_rate)

    start_time = time.time()
    print("Loading data............")
    #
    d = GroupDataset(data_dir=dataset_dir,dataset = args.dataset, config=config)
    print("")
    print("Training data set preprocessing：.....")

    # Link prediction task.......................
    experiment = Experiment(learning_rate=args.lr, batch_size=args.batch_size,
                            num_iterations=args.num_iterations, dim=args.dim,
                            cuda=config.device, nneg=args.nneg, model=args.model)
    experiment.train()


    # Node classification task...................
    Node_d = NodeDataset(data_dir=dataset_dir, config=config)
    node_classification(Node_d,config.save_path,config.embed,config)

