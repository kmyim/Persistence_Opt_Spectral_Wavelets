import networkx as nx

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import rankdata
import numpy as np
import pickle
from scipy.linalg import eigh
from playground import *
from models import ModelStats
import numpy.polynomial.chebyshev as cheby


##### neural net parameters #####

# persistence image
expt_name = 'chebyshev20_stats_fix_diag'
cheby_degree = 20

##### directories #####
dataset_name = 'COX2/'
raw = 'raw_datasets/'
processed = 'processed_datasets/'
result_dump = 'ten10folds/' + dataset_name + expt_name + '/' + expt_name + '_'


##### load data #####

graph_list = pickle.load(open(processed + dataset_name + 'networkx_graphs.pkl', 'rb'))
data_len = len(graph_list)

##### training parameters #####
max_epoch = 500
start_sample = 49#start evaluating on the test set at this epoch
sampling_freq = 50 #evaluate on test set every sampling_freq number of epochs after you start sampling
test_size =  data_len // 10

bs = {'DHFR/': 31, 'MUTAG/': 17, 'COX2/': 36, 'IMDB-BINARY/': 90, 'NCI1/': 137}
pd = {'DHFR/': 3, 'MUTAG/': 5, 'COX2/': 3,'NCI1/': 3}
batch_size = bs[dataset_name]
train_batches = np.ceil((data_len-test_size)/batch_size).astype(int)
print('number of batches = ', train_batches, ' batch size = ', batch_size, ' test size ', test_size)

#pickle.dump(coeffs.numpy(), open(result_dump + 'cheby_init.pkl' , 'wb'))

#ema_decay = 0.9

###### preprocess #####
data = []
label = []
mx, mn = -np.inf, np.inf

for i in range(len(graph_list)):
    G = graph_list[i]
    L = nx.normalized_laplacian_matrix(G)
    w, v = eigh(L.todense())
    vandermonde = cheby.chebvander(w.flatten()-1, cheby_degree)/np.sqrt(cheby_degree)
    datum = dict()
    label.append(G.graph['label'])
    A = np.matmul(v**2, vandermonde[:,1:])
    datum['chebyshev'] = torch.from_numpy(A).float()
    if i ==0:
        Alist = A
    else:
        Alist = np.append(Alist, A, axis = 0)
    data.append(datum)

u, s, vh  = np.linalg.svd(Alist)
pds = pd[dataset_name]
principal_dims = np.argpartition(s, -pds)[-pds : ]
principals = torch.from_numpy(vh[principal_dims].T).float()
#print(principals.shape)
#print(s[principal_dims])
#print(s)
torch.manual_seed(0)
rng_state= torch.get_rng_state() #seed init to ensure same initial conditions for each training
eval_model = ModelStats(pds) # evaluation model
coeffs = eval_model.cheby_params.detach()
coeffs /= torch.norm(coeffs)
#print(coeffs)



for i in range(data_len):
    dat = data[i]

    dat['chebyshev'] = torch.matmul(dat['chebyshev'], principals)
    hks = torch.flatten(torch.matmul(dat['chebyshev'], coeffs)).float().detach()
    mx = max(float(torch.max(hks)), mx)
    mn = min(float(torch.min(hks)), mn)
    G = graph_list[i]
    st = simplex_tree_constructor([list(e) for e in G.edges()])
    dat['simplex_tree'] = filtration_update(st, hks.numpy())

    #recompute persistence
    pers = dat['simplex_tree'].persistence(homology_coeff_field = 2)
    del pers


print(mx, mn)
label = torch.tensor(label).float()
print('Finished initial processing')
del graph_list

torch.set_rng_state(rng_state) #fix init state
eval_model = ModelStats(pds)

####### torch random seeds #######
shuffidx = list(range(data_len)) # data indexer

criterion = nn.BCEWithLogitsLoss() #loss function

for run in range(10):
    print('run = ', run)
    np.random.seed(run)
    np.random.shuffle(shuffidx) # randomly choose partition of data into test / fold
    if run == 0:
        start = 9
    else:
        start = 0
    for fold in range(start, 10):
        print ('> fold ', fold)
        test_acc = []
        train_acc = []
        loss_func = []
        test_reg = []
        train_reg = []
        test_bottom = fold * test_size
        test_top = (1+fold) * test_size
        test_indices = shuffidx[test_bottom : test_top]
        train_indices = shuffidx[0:test_bottom] + shuffidx[test_top :]

        #regulariser_constant = 0.5#reinit regulariser_constant
        torch.set_rng_state(rng_state) #fix init state
        pht = ModelStats(pds, square = [mn, mx])
        #ema.register(eval_model.state_dict(keep_vars = False))

        criterion = nn.BCEWithLogitsLoss() #loss function
        #eval_crit = nn.BCEWithLogitsLoss()

        #optimizer_classifier = optim.Adam([{'params': [param for name, param in pht.named_parameters() if 'cheby_params' not in name]}], lr=1e-3, weight_decay = 0.0)
        #optimizer_filter = optim.Adam([{'params': pht.cheby_params}], lr=1e-2, weight_decay = 0.0 )
        optimizer = optim.Adam(pht.parameters(), lr = 1e-3)
        for epoch in range(max_epoch):
            pht.train()
            np.random.shuffle(train_indices)
            #if epoch % 50 < 20:
            #    pht.error_tol = 0
            #    regulariser_constant = max(regulariser_constant*0.975, 0.1)
            #else:
            #    pht.error_tol = np.inf

            for b in range(train_batches):

                train_indices_batch = train_indices[b*batch_size : (b+1)*batch_size ]
                optimizer.zero_grad()
                #optimizer_classifier.zero_grad()
                #optimizer_filter.zero_grad()

                #outputs, regulariser = pht([data[i] for i in train_indices_batch])
                outputs = pht([data[i] for i in train_indices_batch])
                #loss = criterion(outputs, label[train_indices_batch].unsqueeze(-1)) + regulariser_constant*regulariser
                loss = criterion(outputs, label[train_indices_batch].unsqueeze(-1))
                loss.backward()
                #nn.utils.clip_grad_norm_(pht.cheby_params, 0.05)
                optimizer.step()
                #optimizer_classifier.step()
                #optimizer_filter.step()
            #    if epoch % 50 >= 20:
            #        pht.error_tol = np.inf
            #        optimizer_classifier.step()
            #    else:
            #        optimizer_filter.step()
                ##### compute moving average #####
            #    pht.eval()
            #    ema.update(pht.state_dict(keep_vars = False))
            #    pht.train()

            if epoch >= start_sample and (epoch - start_sample) % sampling_freq == 0:

                #eval_model.load_state_dict(ema.shadow)
                #eval_model.eval()
                pht.eval()
                #test_outputs, tt_reg = pht([data[i] for i in test_indices])
                #train_outputs, tn_reg = pht([data[i] for i in train_indices])
                test_outputs = pht([data[i] for i in test_indices])
                train_outputs =pht([data[i] for i in train_indices])

                test_acc.append(int(((test_outputs.view(-1) > 0) == label[test_indices]).sum())/len(test_indices))
                loss_func.append(criterion(train_outputs, label[train_indices].unsqueeze(-1)).item())
                train_acc.append(int(((train_outputs.view(-1) > 0) == label[train_indices]).sum())/len(train_indices))
                #test_reg.append(float(tt_reg.data))
                #train_reg.append(float(tn_reg.data))
                #print(loss_func[-1], train_acc[-1], test_acc[-1])

        #pht.eval()
        # saves the final learnt filter #
        run_fold_index = str(run) + '_' + str(fold)  + '.pkl'
        saverunfoldfilter = result_dump + 'learnt_filter_' + run_fold_index
        #pickle.dump(eval_model.ev_function(torch.linspace(0,2, 100).unsqueeze(1)).squeeze(1).detach().numpy(), open(saverunfoldfilter , 'wb')) #sampled from lambda = 0... 2 at 100 evenly spaced points
        pickle.dump(pht.cheby_params.detach().numpy(), open(saverunfoldfilter , 'wb')) #sampled from lambda = 0... 2 at 100 evenly spaced points

        pickle.dump(train_acc, open(result_dump + 'train_acc_' + run_fold_index, 'wb'))
        pickle.dump(test_acc, open(result_dump + 'test_acc_' + run_fold_index, 'wb'))
        pickle.dump(loss_func, open(result_dump + 'loss_' + run_fold_index, 'wb'))
        #pickle.dump(train_reg, open(result_dump + 'train_reg_' + run_fold_index, 'wb'))
        #pickle.dump(test_reg, open(result_dump + 'test_reg_' + run_fold_index, 'wb'))
