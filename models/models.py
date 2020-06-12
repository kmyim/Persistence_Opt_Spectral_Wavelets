import torch
import copy
import gudhi
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pickle

from playground import *

def normalize_nzs(x):
    entries = torch.where(x>0)
    nzs = x[entries]
    return torch.mean(nzs), torch.std(nzs)

class PersistenceImage(nn.Module):

    def __init__(self, res, sigma, square = [0.0, 1.0]):

        super(PersistenceImage, self).__init__()

        self.sigma = sigma
        self.resolution = res

        self.x = torch.linspace(square[0], square[1], self.resolution)
        self.y = torch.linspace(square[0], square[1], self.resolution)


    def forward(self, b, d):
        nz = torch.abs(d-b)>0
        X = (b.unsqueeze(-1) - self.x)**2
        X = torch.exp(-X/2/self.sigma**2).transpose(-1,-2)
        Y = (d.unsqueeze(-1) - self.y)**2
        Y = torch.exp(-Y/2/self.sigma**2)
        Y = nz.unsqueeze(-1)*Y
        #PI = torch.matmul(X, Y)/self.sigma**2
        PI = torch.matmul(X, Y)/self.sigma**2
        #X = (b.unsqueeze(-1) - self.x)**2
        #Y = (p.unsqueeze(-1) - self.y)**2
        #D = X.unsqueeze(-1) + Y.unsqueeze(-2)
        #tents = torch.sum(p.unsqueeze(-1).unsqueeze(-1)*self.nonlin(1.0-D/self.sigma), dim = -3) #define self nonlin in class definition as self.nonlin = nn.ReLU(); weight by persistence

        return PI



def linear_block(in_f, out_f, bn = True):

    if bn:
        return nn.Sequential(nn.Linear(in_f, out_f),  nn.BatchNorm1d(out_f), nn.PReLU())
    else:
        return nn.Sequential(nn.Linear(in_f, out_f), nn.PReLU())



class ChebyshevWavelets(nn.Module):

    '''
    initial parameters
    '''

    def __init__(self, cheby_degree = 6, max_intervals = 25):

        super(ChebyshevWavelets, self).__init__()

        self.cheby_params = torch.nn.Parameter(torch.empty([cheby_degree, 1]).normal_(mean = 0, std = 1/np.sqrt(cheby_degree*np.pi/2)), requires_grad = True)
        self.max_num_intervals = max_intervals

    def forward(self, mb):

        L = len(mb)
        births = torch.zeros([L, 3, self.max_num_intervals])
        deaths = torch.zeros([L, 3, self.max_num_intervals])

        for i in range(L):

            datum = mb[i]
            f = torch.matmul(datum['chebyshev'], self.cheby_params).flatten()/torch.norm(self.cheby_params, p=2)
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
                b = f[birthv[k]]
                d = f[deathv[k]]
                if len(b) > self.max_num_intervals:
                    p = torch.abs(f[birthv[k]] - f[deathv[k]])
                    truncation = torch.topk(p, self.max_num_intervals[k], largest=True, sorted=False).indices
                    b = b[truncation]
                    d = d[truncation]
                births[i, k,0:len(b)] = b
                deaths[i, k,0:len(d)] = d
                del b, d
            del birthv, deathv

        return births, deaths




class ModelChebyshev(nn.Module):

    def __init__(self, resolution = 20, sigma = 0.1, error_tol = 0, cheby_degree = 11, max_intervals = 25, square = [0.0, 1.0]):

        super(ModelChebyshev, self).__init__()

        self.barcode = ChebyshevWavelets(cheby_degree, max_intervals)
        self.square = square
        self.sigma = sigma
        self.resolution = resolution
        self.PI = PersistenceImage(self.resolution, self.sigma, self.square)
        self.averager = nn.Sequential(nn.Conv2d(3, 20, 2, 1), nn.BatchNorm2d(20)) #2*20*20 -> 2*19*19
        dimension = int((self.resolution -  1))
        self.final_fc = feed_forward_mlps((20*dimension**2, 20))
        self.final_final_fc = nn.Sequential(nn.Linear(20, 1), nn.BatchNorm1d(1))
        self.update_counter = 0
        self.error_tol = error_tol

    def forward(self, mb):

        L = len(mb)
        births, deaths,  = self.barcode(mb)
        births = torch.reshape(births, [L*3, -1])
        deaths = torch.reshape(deaths, [L*3, -1])
        PIs = self.PI(births, deaths)
        PIs = torch.reshape(PIs, [L, 3, self.resolution, self.resolution])
        del births, deaths

        x = self.averager(PIs)
        x = self.final_fc(x.flatten(1))
        x = self.final_final_fc(x)

        return x
