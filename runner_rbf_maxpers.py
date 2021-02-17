import networkx as nx
import gc
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle
from scipy.linalg import eigh
from playground import *
from models import MP, EMA
import matplotlib.pyplot as plt


##### neural net parameters #####
gc.collect()
expt_name = 'rbf12_maxH0H1rel'
param = 'rbf'
bfs = 12

lr = 1e-1

max_epoch = 50

if param == 'rbf':
    bfseps = 2/(bfs-3)
    centroids = torch.linspace(-bfseps, 2 + bfseps, bfs)


##### directories #####
dataset_name = 'IMDB-BINARY/'
raw = 'raw_datasets/'
processed = 'data_example/'
result_dump = 'ten10folds/' + dataset_name + expt_name + '/' + expt_name + '_'


##### load data #####

graph_list = pickle.load(open(processed + dataset_name + 'networkx_graphs.pkl', 'rb'))
data_len = len(graph_list)

##### training parameters #####
#bs = {'DHFR/': 11, 'MUTAG/': 10, 'COX2/': 9, 'IMDB-BINARY/': 18, 'NCI1/': 20, 'IMDB-MULTI/': 27 }
bs = {'DHFR/': 59, 'MUTAG/': 57, 'COX2/': 47, 'IMDB-BINARY/': 50, 'NCI1/': 20, 'IMDB-MULTI/': 27 }

batch_size = bs[dataset_name]
test_size =  data_len // 10
train_batches = np.ceil((data_len-test_size)/batch_size).astype(int)
print('number of batches = ', train_batches, ' batch size = ', batch_size)

torch.manual_seed(0)
rng_state= torch.get_rng_state() #seed init to ensure same initial conditions for each training

### eigenvalue path signature ####



###### preprocess #####
data = []
Alist = None
Aker = None
smax = []
for i in range(len(graph_list)):
    G = graph_list[i]
    L = nx.normalized_laplacian_matrix(G)
    lam, v = eigh(L.todense())
    w = torch.from_numpy(lam).float()
    diameter = nx.diameter(G)
    datum = dict()

    evecssq = torch.from_numpy(v**2).float()
    if param == 'rbf':
        gram = 1/torch.sqrt((torch.reshape(w, [len(G), 1]) - centroids)**2/bfseps**2 + 1)
        A = torch.matmul(evecssq, gram)
        #A = A - torch.mean(A, dim = 0)

    elif param == 'cheb':
        vander=  torch.from_numpy(np.polynomial.chebyshev.chebvander(lam, bfs+1)).float()
        A = torch.matmul(evecssq, vander[:, 2:])

    datum['diameter'] = diameter
    st = simplex_tree_constructor_ord([list(e) for e in G.edges()])
    datum['simplex_tree'] = filtration_update_ord(st,np.linspace(0, 1, len(G)))
    datum['f'] = torch.linspace(0, 1, len(G))
    pers = datum['simplex_tree'].persistence(homology_coeff_field = 2)

    if diameter > 1:
        if type(Alist) == type(None):
            Alist = A
        else:
            Alist = np.append(Alist, A, axis = 0)

    data.append(datum)

u, s, vh  = np.linalg.svd(Alist, full_matrices = False)

del Alist
gc.collect()
print('u shape, ', u.shape)
print('s ', s)

torch.set_rng_state(rng_state)
torch.manual_seed(0)

cnter = 0
for i in range(len(graph_list)):
    dat = data[i]
    if dat['diameter'] > 1:
        G = graph_list[i]
        dat['secondary_gram'] = torch.from_numpy(u[cnter:cnter + len(G),:])
        cnter+= len(G)

del u, s, vh

gc.collect()

print('Finished initial processing')
del graph_list
gc.collect()

torch.set_rng_state(rng_state) #fix init state
shuffidx = list(range(data_len)) # data indexer


for run in range(10):
    print('run = ', run)
    np.random.seed(run)
    np.random.shuffle(shuffidx) # randomly choose partition of data into test / fold

    for fold in range(10):
        print ('> fold ', fold)

        test_bottom = fold * test_size
        test_top = (1+fold) * test_size
        test_indices = shuffidx[test_bottom : test_top]
        train_indices = shuffidx[0:test_bottom] + shuffidx[test_top :]
        loss_func = []
        bfs_params = []
        test_loss = []

        torch.set_rng_state(rng_state) #fix init state
        torch.manual_seed(0)
        pht = MP(rbf = bfs, filtername = 'f')

        optimizer_filter = optim.SGD([{'params': pht.rbfweights}], lr=lr)

        for epoch in range(max_epoch):
            pht.train()
            np.random.shuffle(train_indices)
            lss = 0

            for b in range(train_batches):

                bfs_params.append(pht.rbfweights.detach().clone().numpy())
                pht.train()
                train_indices_batch = train_indices[b*batch_size : (b+1)*batch_size ]
                optimizer_filter.zero_grad()

                loss = pht([data[i] for i in train_indices_batch])

                loss.backward()
                optimizer_filter.step()
                lss += float(loss)

            loss_func.append(float(lss)/len(train_indices))

            pht.eval()
            test_lss = pht([data[i] for i in test_indices])
            test_loss.append(float(test_lss) / len(test_indices))


        run_fold_index = str(run) + '_' + str(fold)
        pickle.dump(test_loss, open(result_dump + 'test_loss_' + run_fold_index + '.pkl', 'wb'))
        pickle.dump(loss_func, open(result_dump + 'train_loss_' + run_fold_index+'.pkl', 'wb'))
        pickle.dump(bfs_params, open(result_dump + 'theta_' + run_fold_index+ '.pkl', 'wb'))

        del bfs_params, test_lss, loss_func, pht

        gc.collect()
