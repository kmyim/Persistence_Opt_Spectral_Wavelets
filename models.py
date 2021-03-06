import torch
import copy
import gudhi
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import multiprocessing
from playground import *
import iisignature
from multiprocessing import Pool

from functools import partial

def normalize_nzs(x):
    entries = torch.where(x>0)
    nzs = x[entries]
    return torch.mean(nzs), torch.std(nzs)


class PersistenceLandscapes(nn.Module):

    def __init__(self, res, layers):

        super(PersistenceLandscapes, self).__init__()

        self.resolution = res
        self.t = torch.linspace(0,1, self.resolution)[0:self.resolution-1]
        self.plus = nn.ReLU()
        self.layers = layers

    def forward(self, b, d):

        tents = torch.max(b.unsqueeze(-1) - self.t , self.t - d.unsqueeze(-1))
        tents = self.plus(tents)
        ls = torch.topk(tents, self.layers, axis = 1, largest  = True, sorted = True)
        return ls.values

class SlicedWasserstein(nn.Module):

    def __init__(self, res):

        super(SlicedWasserstein, self).__init__()

        self.resolution = res
        theta = np.pi*torch.linspace(0,1, self.resolution+2)[1:self.resolution+1]
        self.x = torch.cos(theta)
        self.y = torch.sin(theta)

    def forward(self, b, d):
        x = self.x*b.unsqueeze(-1) + self.y*d.unsqueeze(-1)
        return torch.sort(x, axis = -2).values

class PersistenceImage(nn.Module):

    def __init__(self, res, sigma, square = [0.0, 1.0]):

        super(PersistenceImage, self).__init__()

        self.sigma = sigma
        self.resolution = res
        self.range = abs(square[1] - square[0])
        #self.x = torch.linspace(square[0], square[1], self.resolution)
        #self.y = torch.linspace(square[0], square[1], self.resolution) # d version
        self.offset = self.range/(self.resolution - 3)
        self.x = torch.linspace(square[0]-self.offset, square[1] + self.offset, self.resolution)

        self.y = torch.linspace(-self.offset, self.range + self.offset, self.resolution)
        self.r = 2

    def forward(self, b, d):
        nz = torch.abs(d-b)>self.offset/2
        N = int(torch.sum(nz))
        if N > 0:
            p = torch.abs(d-b)
            X = (b.unsqueeze(-1) - self.x)**2
            X = torch.exp(-X/2/self.sigma**2).transpose(-1,-2)
            #Y = (d.unsqueeze(-1) - self.y)**2
            Y =  (p.unsqueeze(-1) - self.y)**2
            Y = torch.exp(-Y/2/self.sigma**2)
            Y = nz.unsqueeze(-1)*Y
            #Y = (p**2).unsqueeze(-1)*Y
            #w = 1-torch.cos(np.pi*p/self.range)
            w  = torch.clamp(p, 0,self.offset)*np.pi/self.offset
            w = (1- torch.cos(w))/2
            Y = w.unsqueeze(-1)*Y
            PI = torch.matmul(X, Y)/self.sigma**2
        else:
            PI = torch.zeros([self.resolution, self.resolution])

        # exp
        #X = torch.abs(b.unsqueeze(-1) - self.x)
        #X = torch.exp(-X/self.sigma).transpose(-1,-2)
        #Y = torch.abs(d.unsqueeze(-1) - self.y)
        #Y = torch.exp(-Y/self.sigma)
        #PI = torch.matmul(X, Y)/self.sigma


        #X = torch.abs(b.unsqueeze(-1) - self.x)
        #Y = torch.abs(d.unsqueeze(-1) - self.y)
        #D = nz.unsqueeze(-1).unsqueeze(-1)*(X.unsqueeze(-1) + Y.unsqueeze(-2))/self.sigma
        #tents = torch.sum(nz.unsqueeze(-1).unsqueeze(-1)*self.nonlin(1.0-D/self.sigma), dim = -3) #define self nonlin in class definition as self.nonlin = nn.ReLU(); weight by persistence
        #Dtrace = torch.sum(D, axis = 1)
        #PI = (1/(1+Dtrace) - 1/(1+ torch.abs(self.r-Dtrace)))*(1/self.r+1)
        #print(PI.shape)
        #tents = torch.sum(p.unsqueeze(-1).unsqueeze(-1)*self.nonlin(1.0-D/self.sigma), dim = -3) #define self nonlin in class definition as self.nonlin = nn.ReLU(); weight by persistence

        return PI

class DiagramStats(nn.Module):

    def __init__(self):

        super(DiagramStats, self).__init__()

    def forward(self, b, d):

        features = torch.zeros([b.shape[0], 6]) #new feats
        #features = torch.zeros([b.shape[0], 5])
        p = torch.abs(d-b)
        logp = torch.log(1+p)

        features[:,0] = torch.sum(p, axis = -1)
        features[:,1] = torch.sum(b*p, axis = -1)
        features[:,2] = torch.sum(d*p, axis = -1)
        #features[:,3] = torch.sum((b**2)*(p**4), axis = -1)
        #features[:,4] = torch.sum((d**2)*(p**4), axis = -1)
        ##features[:,3] = torch.log(1.0 + torch.sum(torch.exp(-b)*(p>0), axis = -1))
        #features[:,4] = torch.log(torch.exp(torch.tensor(-1.0)) + torch.sum(torch.exp(d - 1)*(p>0), axis = -1)) + 1
        ## new feats
        features[:,3] = torch.sum(b*logp, axis = -1)
        features[:,4] = torch.sum(d*logp, axis = -1)
        features[:,5] = torch.log(torch.exp(torch.tensor(-1.0)) + torch.sum(torch.exp(p - 1)*(p>0), axis = -1) ) + 1

        return features

def linear_block(in_f, out_f, bn = True):

    if bn:
        #return nn.Sequential(nn.Linear(in_f, out_f),  nn.BatchNorm1d(out_f), nn.ReLU())
        return nn.Sequential(nn.Linear(in_f, out_f), nn.ReLU(),  nn.BatchNorm1d(out_f))
    else:
        return nn.Sequential(nn.Linear(in_f, out_f), nn.ReLU())

class ModelPIRBF(nn.Module):

    def __init__(self, rbf = 10, resolution = 10, lims = [0.0, 1.0], weightinit = None):

        super(ModelPIRBF, self).__init__()

        if weightinit == None:
            self.rbfweights = nn.Parameter(torch.empty(rbf, 1).normal_(mean = 0, std = 1/np.sqrt(rbf)), requires_grad = True)
        else:
            self.rbfweights = nn.Parameter(torch.reshape(weightinit, [rbf, 1]), requires_grad = True)
        #self.rbfweights = nn.Parameter(torch.empty(rbf, 1).uniform_(-1,1), requires_grad = True)
        self.resolution = resolution
        #self.PI = PersistenceImage(self.resolution,1/(resolution+1))
        #self.CNN = nn.Sequential(nn.Conv2d(3, 15, 2, 1, 1), nn.ReLU(), nn.BatchNorm2d(15, affine = False))
        self.max_num_intervals = [100, 100, 100]
        #butt_res =15*(self.resolution+1)**2
        #self.project =nn.Sequential(nn.Linear(butt_res, 1))
        self.lims = lims
        #self.CNN = nn.BatchNorm2d(3, affine = False)
        #self.project = nn.Sequential(nn.Linear(3*self.resolution**2, 1) )
        self.sigma = 1/(resolution-3)
        self.resolution = resolution
        self.PI = PersistenceImage(self.resolution, self.sigma)
        self.CNN= nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(in_channels = 3,out_channels = 15, kernel_size = 2, stride = 1), nn.ReLU(), nn.BatchNorm2d(15), nn.Conv2d(in_channels = 15,out_channels = 1, kernel_size = 2, stride = 1), nn.ReLU(), nn.Dropout2d(0.6))
        #butt = ((self.resolution // 2 -1)//2 -1)**2
        butt = (self.resolution -2)**2

        self.project = nn.Linear(butt, 1)
        self.mn = min(lims)
        self.mx = max(lims)
        self.range = abs(self.mx - self.mn)

        #self.CNN= nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(in_channels = 3,out_channels = 1, kernel_size = 2, stride = 1), nn.ReLU(), nn.BatchNorm2d(1))
        #butt = (self.resolution -1)**2
        self.project = nn.Linear(butt,1)

        self.freeze_persistence = False
        self.update = False

    def forward(self, mb):
        L = len(mb)
        births = torch.zeros([L, 3, max(self.max_num_intervals)])
        deaths = torch.zeros([L, 3, max(self.max_num_intervals)])

        if self.freeze_persistence == True:
            PIs = torch.zeros([L, 3, self.resolution, self.resolution])
            for i in range(L):
                PIs[i] = mb[i]['images']
        else:

            for i in range(L):
                datum = mb[i]
                f = (torch.matmul(datum['secondary_gram'], self.rbfweights).flatten() - self.mn)/(self.range)

                birthv = [[],[],[]]
                deathv = [[],[],[]]
                #recompute persistence
                datum['simplex_tree'] = filtration_update(datum['simplex_tree'], f.detach().numpy())
                pers = datum['simplex_tree'].persistence(homology_coeff_field = 2)

                pairs =  datum['simplex_tree'].persistence_pairs()
                for interval in pairs:
                    if len(interval[1]) == 0: #skip infinite bars
                        continue
                    else:
                        if len(interval[0]) == 1: #H0
                            bv = interval[0][0]
                            dv = max([v for v in interval[1] if v != 1000], key = lambda v: f[v])
                            k = 0
                        if len(interval[0]) == 2: #H1
                            dv = max([v for v in interval[0] if v != 1000], key = lambda v: f[v])
                            bv = min([v for v in interval[1] if v != 1000], key = lambda v: f[v])
                            if 1000 in interval[0]:
                                k = 1
                            else:
                                k = 2
                    birthv[k].append(bv)
                    deathv[k].append(dv)

                for k in range(3):
                    if len(birthv[k]) > 0:
                        b = f[birthv[k]]
                        d = f[deathv[k]]

                        if len(b) > self.max_num_intervals[k]:
                            p = torch.abs(f[birthv[k]] - f[deathv[k]])
                            truncation = torch.topk(p, self.max_num_intervals[k], largest=True, sorted=False).indices
                            b = b[truncation]
                            d = d[truncation]
                        births[i, k,0:len(b)] = b
                        deaths[i, k,0:len(d)] = d
                        del b, d
                del birthv, deathv

            births = torch.reshape(births, [L*3, -1])
            deaths = torch.reshape(deaths, [L*3, -1])
            PIs = self.PI(births, deaths)
            PIs = torch.reshape(PIs, [L, 3, self.resolution, self.resolution])

            if self.update :
                for i in range(L):
                    mb[i]['images'] = PIs[i].detach().clone().unsqueeze(0)
            del births, deaths
        PIs = self.CNN(PIs)
        features = torch.reshape(PIs, [L, -1])

        return self.project(features)

class SigFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, m):
        result = iisignature.sig(X.detach().numpy(), m)
        ctx.save_for_backward(X, torch.tensor(m))
        return torch.tensor(result).float()

    @staticmethod
    def backward(ctx, grad_output):
        X, m,  = ctx.saved_tensors
        result = iisignature.sigbackprop(grad_output.detach().numpy(), X.detach().numpy(), int(m))
        return torch.tensor(result).float(), None

class LogSigFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, s):
        result = iisignature.logsig(X.detach().numpy(),s)
        ctx.save_for_backward(X)
        ctx.s = s
        return torch.tensor(result).float()

    @staticmethod
    def backward(ctx, grad_output):
        X,  = ctx.saved_tensors
        result = iisignature.logsigbackprop(grad_output.detach().numpy(), X.detach().numpy(), ctx.s)
        return torch.tensor(result).float(), None

class feed_forward_mlps(nn.Module):

    def __init__(self, layer_dims, bn = True):

        super(feed_forward_mlps, self).__init__()

        mlp_list = []

        for l in range(len(layer_dims)-1):
            mlp_list.append(linear_block(layer_dims[l], layer_dims[l+1], bn))

        self.sequenced = nn.Sequential(*mlp_list)

    def forward(self, x):

        return self.sequenced(x)

class cnn_builder(nn.Module):

    def __init__(self, pxs, channels, kernel, stride, padding, groups, dropout):

        super(cnn_builder, self).__init__()

        mlp_list = []
        width = pxs

        for l in range(len(channels)-1):
            width = int(np.floor((width - kernel[l] + 2*padding[l])/stride[l]) + 1)

            if dropout and l == len(channels)-2 :

                mlp_list.append(nn.Sequential(nn.BatchNorm2d(channels[l], affine = False), nn.Dropout2d(0.5), nn.Conv2d(in_channels = channels[l], out_channels = channels[l+1], kernel_size = kernel[l], stride = stride[l], groups = groups[l], padding = padding[l],), nn.ReLU()))

            else:
                mlp_list.append(nn.Sequential(nn.BatchNorm2d(channels[l], affine = False), nn.Conv2d(in_channels = channels[l],out_channels = channels[l+1], kernel_size = kernel[l], stride = stride[l], groups = groups[l], padding = padding[l],), nn.ReLU()))


        if not dropout:
            mlp_list.append(nn.BatchNorm2d(channels[-1], affine = False))

        self.sequenced = nn.Sequential(*mlp_list)
        self.out_size = channels[-1]*width**2

    def forward(self, x):

        return self.sequenced(x)

class EMA():

    def __init__(self, alpha):

        self.alpha = alpha
        self.shadow = dict()


    def register(self, state_dict):
        self.shadow = state_dict

    def update(self, new_state_dict):
        for name in self.shadow:
            if not isinstance(new_state_dict[name], str):
                self.shadow[name] = (1-self.alpha)*new_state_dict[name] + self.alpha*self.shadow[name]

class ModelPIRBFDoubleOneStatic(nn.Module):

    def __init__(self, rbf = 10, resolution = 10, lims = [0.0, 1.0], weightinit = None, extra_feat_len = 0):

        super(ModelPIRBFDoubleOneStatic, self).__init__()

        if type(weightinit) == type(None):
            self.rbfweights = nn.Parameter(torch.empty(rbf, 1).normal_(mean = 0, std = 1/np.sqrt(rbf)), requires_grad = True)
        else:
            self.rbfweights = nn.Parameter(torch.reshape(weightinit.detach().clone(), [rbf, 1]), requires_grad = True)
        self.resolution = resolution
        self.max_num_intervals = [500, 500, 500]
        self.lims = lims
        self.sigma = 1/(resolution-3)
        self.resolution = resolution
        self.filtername = 'f'
        self.PHPI = GenPHandPI(self.resolution, self.lims, self.max_num_intervals, self.filtername)
        channels = [6,20,2]
        kernel = [2,2]
        stride = [1,1]
        padding = [1,1]
        groups = [2,2]
        dropout = True
        #self.CNN= nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(in_channels = 3,out_channels = 15, kernel_size = 2, stride = 1), nn.ReLU(), nn.BatchNorm2d(15), nn.Conv2d(in_channels = 15,out_channels = 1, kernel_size = 2, stride = 1), nn.ReLU(), nn.Dropout2d(0.6))
        #self.CNN= nn.Sequential(nn.BatchNorm2d(6), nn.Conv2d(in_channels = 6,out_channels = 16, kernel_size = 2, stride = 1, groups = 2), nn.ReLU(), nn.BatchNorm2d(16), nn.Conv2d(in_channels = 16,out_channels = 2, kernel_size = 2, stride = 1), nn.ReLU())
        #self.CNN= nn.Sequential(nn.BatchNorm2d(conv_filters[0]), nn.Conv2d(in_channels = conv_filters[0],out_channels = conv_filters[1], kernel_size = kern_size[0], stride = stride[0], groups = 2), nn.ReLU(), nn.BatchNorm2d(conv_filters[1]), nn.Conv2d(in_channels = conv_filters[1],out_channels = conv_filters[2], kernel_size = kern_size[1], stride = stride[1], groups = 2), nn.ReLU(),  nn.BatchNorm2d(conv_filters[2]))
        self.CNN = cnn_builder(self.resolution, channels, kernel, stride, padding, groups, dropout)

        self.extra_feat_len = extra_feat_len
        #butt = self.CNN.out_size + extra_feat_len
        butt = self.CNN.out_size + self.extra_feat_len

        if extra_feat_len > 0:
            self.feats_preprocess = nn.Sequential(nn.BatchNorm1d(self.extra_feat_len, affine = False), nn.Linear(self.extra_feat_len, self.extra_feat_len), nn.ReLU(), nn.BatchNorm1d(self.extra_feat_len, affine = False))

        self.project = nn.Sequential(nn.Linear(butt, 1))
        self.mn = min(lims)
        self.mx = max(lims)
        self.range = abs(self.mx - self.mn)

        self.freeze_persistence = False
        self.update = False

    def forward(self, mb):

        L = len(mb)

        if self.extra_feat_len > 0:
            feats = torch.cat([datum['feats'].unsqueeze(0) for datum in mb], dim = 0)


        if self.freeze_persistence:
            PIs = torch.cat([datum['images'].unsqueeze(0) for datum in mb], dim = 0)
        #minmax = torch.cat([datum['fminmax'] for datum in mb], dim = 0)
        else:
            #minmax = torch.zeros([L, 2])
            PI_static =  torch.cat([datum['images'][3:].unsqueeze(0) for datum in mb], dim = 0)

            for i in range(L):
                if mb[i]['diameter'] > 1:
                    f = torch.matmul(mb[i]['secondary_gram'], self.rbfweights).flatten()
                    mb[i][self.filtername] = f
                #minmax[i,0] = torch.min(f)
                #minmax[i,1] = torch.max(f)
                #if self.update :
                #    mb[i]['fminmax'] = minmax[i].detach().clone().unsqueeze(0)

            PI_dynamic = self.PHPI(mb)
            PIs = torch.cat([PI_dynamic, PI_static], dim = 1)
            if self.update :
                for i in range(L):
                    if mb[i]['diameter'] > 1:
                        mb[i]['images'][0:3] = PI_dynamic[i].detach().clone().unsqueeze(0)

        PIs = self.CNN(PIs)
        features = torch.reshape(PIs, [L, -1])

        if  self.extra_feat_len > 0:
            #feats = torch.cat((feats, minmax), dim = 1)
            feats = self.feats_preprocess(feats)
            features = torch.cat((features, feats), dim = 1)

        return self.project(features)

class GenPHandPI(nn.Module):

    def __init__(self, resolution = 10, lims = [0.0, 1.0], max_num_intervals = [100, 100, 100], filtername = 'f'):

        super(GenPHandPI, self).__init__()

        self.resolution = resolution
        self.max_num_intervals = max_num_intervals
        self.mn = min(lims)
        self.mx = max(lims)
        self.range = abs(self.mx - self.mn)

        self.sigma = 1/(resolution-3)
        self.resolution = resolution
        self.PI = PersistenceImage(self.resolution, self.sigma)
        self.filtername = filtername



    def forward(self, mb):
        L = len(mb)
        births = torch.zeros([L, 3, max(self.max_num_intervals)])
        deaths = torch.zeros([L, 3, max(self.max_num_intervals)])

        for i in range(L):
            f = (mb[i][self.filtername]- self.mn)/self.range
            if torch.max(f) - torch.min(f) > self.sigma/10 and mb[i]['diameter'] > 1:
                bv, dv = critical_vertices(mb[i], self.max_num_intervals, self.filtername )

                for k in range(3):
                    l = len(bv[k])
                    if l > 0:
                        births[i, k, 0:l] = f[bv[k]]
                        deaths[i, k, 0:l] = f[dv[k]]


        births = torch.reshape(births, [L*3, -1])
        deaths = torch.reshape(deaths, [L*3, -1])
        PIs = self.PI(births, deaths)
        PIs = torch.reshape(PIs, [L, 3, self.resolution, self.resolution])
        del births, deaths

        return PIs

def critical_vertices(datum, max_num_intervals= [100,100,100], filtername = 'f'):

    f = datum[filtername]

    birthv = [[],[],[]]
    deathv = [[],[],[]]
    #recompute persistence
    datum['simplex_tree'] = filtration_update(datum['simplex_tree'], f.detach().numpy())
    pers = datum['simplex_tree'].persistence(homology_coeff_field = 2)

    pairs =  datum['simplex_tree'].persistence_pairs()
    for interval in pairs:
        if len(interval[1]) == 0: #skip infinite bars
            continue
        else:
            if len(interval[0]) == 1: #H0
                bv = interval[0][0]
                dv = max([v for v in interval[1] if v != 1000], key = lambda v: f[v])
                k = 0
            if len(interval[0]) == 2: #H1
                dv = max([v for v in interval[0] if v != 1000], key = lambda v: f[v])
                bv = min([v for v in interval[1] if v != 1000], key = lambda v: f[v])
                if 1000 in interval[0]:
                    k = 1
                else:
                    k = 2
        if torch.abs(f[bv] - f[dv])> 1e-9:
            birthv[k].append(bv)
            deathv[k].append(dv)

    for k in range(3):
        if len(birthv[k]) > max_num_intervals[k]:
            p = torch.abs(f[birthv[k]] - f[deathv[k]])
            truncation = torch.topk(p, max_num_intervals[k], largest=True, sorted=False).indices
            birthv[k] = [birthv[k][i] for i in truncation]
            deathv[k] = [deathv[k][i] for i in truncation]
    return birthv, deathv


class ModelPIRBFDouble(nn.Module):

    def __init__(self, rbf = 10, resolution = 10, lims = [0.0, 1.0], weightinit = None, extra_feat_len = 0):
        '''
        Now weight_init is a 2x #params tensor
        lims is a 2x2 list

        '''

        super(ModelPIRBFDouble, self).__init__()

        if type(weightinit) == type(None):
            self.rbfweights = nn.Parameter(torch.empty(rbf, 2).normal_(mean = 0, std = 1/np.sqrt(rbf)), requires_grad = True)
        else:
            self.rbfweights = nn.Parameter(torch.reshape(weightinit.detach().clone(), [rbf, 2]), requires_grad = True)
        self.resolution = resolution
        self.max_num_intervals = [500, 500, 500]
        self.lims = lims
        self.sigma = 1/(resolution-3)
        self.resolution = resolution
        self.filtername = ['f', 'g']
        self.PHPI1 = GenPHandPI(self.resolution, self.lims[0], self.max_num_intervals, self.filtername[0])
        self.PHPI2 = GenPHandPI(self.resolution, self.lims[1], self.max_num_intervals, self.filtername[1])
        channels = [6, 20, 2]
        kernel = [2,2]
        stride = [1,1]
        padding = [1,1]
        groups = [2,2]
        dropout = True

        self.CNN = cnn_builder(self.resolution, channels, kernel, stride, padding, groups, dropout)

        self.extra_feat_len = extra_feat_len
        #butt = self.CNN.out_size + extra_feat_len
        butt = self.CNN.out_size + self.extra_feat_len

        if extra_feat_len > 0:
            self.feats_preprocess = nn.Sequential(nn.BatchNorm1d(self.extra_feat_len, affine = False), nn.Linear(self.extra_feat_len, self.extra_feat_len), nn.ReLU(), nn.BatchNorm1d(self.extra_feat_len, affine = False))

        self.project = nn.Sequential(nn.Linear(butt, 1))

        self.freeze_persistence = False
        self.update = False

    def forward(self, mb):

        L = len(mb)

        if self.extra_feat_len > 0:
            feats = torch.cat([datum['feats'].unsqueeze(0) for datum in mb], dim = 0)


        if self.freeze_persistence:
            PIs = torch.cat([datum['images'].unsqueeze(0) for datum in mb], dim = 0)
        #minmax = torch.cat([datum['fminmax'] for datum in mb], dim = 0)
        else:
            #minmax = torch.zeros([L, 2])
            #PI_static =  torch.cat([datum['images'][3:].unsqueeze(0) for datum in mb], dim = 0)

            for i in range(L):
                if mb[i]['diameter'] > 1:
                    for k in range(2):
                        f = torch.matmul(mb[i]['secondary_gram'], self.rbfweights[:,k]).flatten()
                        mb[i][self.filtername[k]] = f
                #minmax[i,0] = torch.min(f)
                #minmax[i,1] = torch.max(f)
                #if self.update :
                #    mb[i]['fminmax'] = minmax[i].detach().clone().unsqueeze(0)

            PI_dynamic1 = self.PHPI1(mb)
            PI_dynamic2 = self.PHPI2(mb)

            PIs = torch.cat([PI_dynamic1, PI_dynamic2], dim = 1)

            #PIs = torch.cat([PI_dynamic1, PI_dynamic2, PI_static], dim = 1)
            if self.update :

                for i in range(L):
                    if mb[i]['diameter'] > 1:
                        mb[i]['images'][0:3] = PI_dynamic1[i].detach().clone().unsqueeze(0)
                        mb[i]['images'][3:6] = PI_dynamic2[i].detach().clone().unsqueeze(0)
        PIs = self.CNN(PIs)
        features = torch.reshape(PIs, [L, -1])

        if  self.extra_feat_len > 0:
            #feats = torch.cat((feats, minmax), dim = 1)
            feats = self.feats_preprocess(feats)
            features = torch.cat((features, feats), dim = 1)

        return self.project(features)





def max_pers(datum, filtername = 'f'):

    f = datum[filtername]

    birthv = [[],[]]
    deathv = [[],[]]
    #recompute persistence
    datum['simplex_tree'] = filtration_update_ord(datum['simplex_tree'], f.detach().numpy())
    pers = datum['simplex_tree'].persistence(homology_coeff_field = 2)
    pairs =  datum['simplex_tree'].persistence_pairs()

    for interval in pairs:
        if len(interval[0]) == 1 and len(interval[1]) > 0: #H0
            bv = interval[0][0]
            dv = max([v for v in interval[1]], key = lambda v: f[v])
            birthv[0].append(bv)
            deathv[0].append(dv)

    datum['simplex_tree'] = filtration_update_ord(datum['simplex_tree'], -f.detach().numpy())
    pers = datum['simplex_tree'].persistence(homology_coeff_field = 2)
    pairs =  datum['simplex_tree'].persistence_pairs()

    for interval in pairs:
        if len(interval[0]) == 1 and len(interval[1]) > 0: #H0
            bv = interval[0][0]
            dv = max([v for v in interval[1]], key = lambda v: f[v])
            birthv[1].append(bv)
            deathv[1].append(dv)

    return birthv, deathv


class MP(nn.Module):

    def __init__(self, rbf = 12, filtername = 'f', weightinit = None  ):

        super(MP, self).__init__()

        weightinit = weightinit

        if type(weightinit) == type(None):
            self.rbfweights = nn.Parameter(torch.empty(rbf, 1).normal_(mean = 0, std = 1/np.sqrt(rbf)), requires_grad = True)
        else:
            self.rbfweights = nn.Parameter(torch.reshape(weightinit.detach().clone(), [rbf, 1]), requires_grad = True)

        self.filtername = filtername

    def forward(self, mb):
        L = len(mb)
        loss = torch.tensor(0.0).float()

        for i in range(L):
            if mb[i]['diameter'] > 1:
                f = torch.matmul(mb[i]['secondary_gram'], self.rbfweights).flatten()/torch.norm(self.rbfweights)
                mb[i][self.filtername] = f
                bv, dv = max_pers(mb[i], self.filtername)
                for k in range(2):
                    loss += torch.sum((f[dv[k]] - f[bv[k]])**2)
        return -loss
