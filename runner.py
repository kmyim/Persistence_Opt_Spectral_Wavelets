import numpy as np
import networkx as nx

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
from scipy.linalg import eigh

import pickle

from models import models
from models import utils
import numpy.polynomial.chebyshev as cheby

cheby_degree = 6
max_intervals = 100

dataset_name = 'COX2'

graph_list = pickle.load(open('data_example/' + dataset_name + '/networkx_graphs.pkl', 'rb'))

print('The ' + dataset_name + ' dataset has ', len(graph_list), ' graphs.')


data = []
for i in range(len(graph_list)):
    
    G = graph_list[i]
    datum = dict()
    L = nx.normalized_laplacian_matrix(G)
    w, v = eigh(L.todense()) #computes eigenvalues w and eigenvectors
    vandermonde = cheby.chebvander(w.flatten()-1, cheby_degree)
    datum['chebyshev'] = torch.from_numpy(np.matmul(v**2, vandermonde[:, 1:])).float()

    hks = np.matmul(v**2,  np.exp(-0.1*w)).flatten() #random initial filtration for the simplex_tree
    st = utils.simplex_tree_constructor([list(e) for e in G.edges()])
    datum['simplex_tree'] = utils.filtration_update(st, hks)
    data.append(datum)
print('Finished initial processing')
del graph_list

data_len = len(data)
test_size = data_len // 10
train_size = data_len - test_size

### training parameters #####
#batch_size = 31 #DHFR
#batch_size = 17 #MUTAG
batch_size = 36 #COX2
#batch_size = 90 #IMDB-BINARY
#batch_size = 137 #NCI1

train_batches = np.ceil((data_len-test_size)/batch_size).astype(int)
max_epoch = 10

print('num points = ', data_len, ' number of batches = ', train_batches, ' batch size = ', batch_size, ' test size ', test_size)

p_tracker = []
tt_loss = []
tn_loss = []
torch.manual_seed(99)
rng_state= torch.get_rng_state() #seed init to ensure same initial conditions for each training

for split in range(10):
    
    np.random.seed(split)
    shuffidx = list(range(data_len)) # data indexer
    np.random.shuffle(shuffidx)

    ####### torch random seeds #######

    for fold in range(10):
        print ('> fold ', fold)

        param_tracker = []
        test_loss = []
        train_loss = []

        test_bottom = fold * test_size
        test_top = (1+fold) * test_size
        test_indices = shuffidx[test_bottom : test_top]
        train_indices = shuffidx[0:test_bottom] + shuffidx[test_top :]

        torch.set_rng_state(rng_state) #fix init state
        barcodes = models.ChebyshevWavelets(cheby_degree = cheby_degree, max_intervals = max_intervals)
        param_tracker.append(list(barcodes.cheby_params.detach().flatten().numpy()))

        optimizer = optim.SGD(barcodes.parameters(), lr=1e-3, weight_decay = 0.0)

        for epoch in range(max_epoch):

            barcodes.train()
            #np.random.shuffle(train_indices)
            for b in range(train_batches):

                train_indices_batch = train_indices[b*batch_size : (b+1)*batch_size ]
                optimizer.zero_grad()
                births, deaths = barcodes([data[i] for i in train_indices])
                loss = -torch.sum((deaths - births)**2)/train_size
                loss.backward()

                optimizer.step()

            barcodes.eval()
            param_tracker.append(list(barcodes.cheby_params.detach().flatten().numpy()))


            barcodes.eval()
            b,d = barcodes([data[i] for i in train_indices])
            tnl = torch.sum((d- b)**2)/train_size
            b,d  = barcodes([data[i] for i in test_indices])
            ttl = torch.sum((d- b)**2)/test_size
            test_loss.append(ttl.detach().numpy())
            train_loss.append(tnl.detach().numpy())

            if epoch == max_epoch-1:
                print(epoch, param_tracker[-1])
                print('train: ', train_loss[-1])
                print('test: ',test_loss[-1])

        p_tracker.append(param_tracker)
        tt_loss.append(test_loss)
        tn_loss.append(train_loss)
        
pickle.dump([tt_loss, tn_loss], open(dataset_name + '_tp2.pkl', 'wb'))