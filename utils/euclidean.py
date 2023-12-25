#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2023/4/20 15:24
# @Author : 慕夏
# @File : euclidean.py
# @Project :KG_GNN_MedFrau


class Euclidean(object):

    def __init__(self):
        super(Euclidean,self).__init__()
        self.name = 'Euclidean'

    def normalize(self, p):
        dim = p.size(-1)
        p.view(-1, dim).renorm_(2, 0, 1.)
        return p

    def sqdist(self, p1, p2, c):
        """
        compute Euclidean distance
        :param p1:
        :param p2:
        :param c: curvalture
        :return: distance value
        """
        return (p1 - p2).pow(2).sum(dim=-1)

    def egrad2rgrad(self, p, dp, c):
        return dp

    def proj(self, p, c):
        return p

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        return p + u

    def logmap(self, p1, p2, c):
        return p2 - p1

    def expmap0(self, u, c):
        return u

    def logmap0(self, p, c):
        return p

    def mobius_add(self, x, y, c, dim=-1):
        """
        general add in Euclidean Space
        :param x:
        :param y:
        :param c:curvalture
        :param dim:
        :return:
        """
        return x + y

    def mobius_matvec(self, m, x, c):
        """
        general matrix multiplication
        :param m:
        :param x:
        :param c:curvalture
        :return:
        """
        mx = x @ m.transpose(-1, -2)
        return mx

    def init_weights(self, w, c, irange=1e-5):
        """
        init weight value using uniform distribution
        :param w:
        :param c:
        :param irange: default 1e-5
        :return:
        """
        w.data.uniform_(-irange, irange)
        return w

    def inner(self, p, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, v, c):
        return v

    def ptransp0(self, x, v, c):
        return x + v
