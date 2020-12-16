import networkx as nx

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pickle
from scipy.linalg import eigh
from playground import *
from models import ModelStatsRBF

import gc


##### neural net parameters #####

error_tol = 0 #only recompute persistence when change in filter function > error_tol
expt_name = 'rbf12_3_stats'
rbf = 12
pds= 3
relabel_p = 0.
smooth_squeeze = 0.


##### directories #####
dataset_name = 'DHFR'
raw = 'raw_datasets/'
processed = 'processed_datasets/'
result_dump = 'ten10folds/' + dataset_name + '/' + expt_name + '/' + expt_name + '_'

##### load data #####

graph_list = pickle.load(open(processed + dataset_name + '/networkx_graphs.pkl', 'rb'))
data_len = len(graph_list)

##### rbf params
#centroids = torch.linspace(0, 2, rbf)
#rbfeps = 2/(rbf+1)
rbfeps = 2/(rbf-3)
centroids = torch.linspace(-rbfeps, 2 + rbfeps, rbf)


#ev = []
#for i in range(len(graph_list)):
#    G = graph_list[i]
#    L = nx.normalized_laplacian_matrix(G)
#    w, v = eigh(L.todense())
#    ev.extend(list(w))
#print(len(ev))
#centroids = np.quantile(a = np.array(ev), q = np.linspace(0, 1, rbf))
#rbfeps = float(np.median(np.diff(centroids)))
#centroids = torch.from_numpy(centroids).float()
#print('rbfeps = ', rbfeps)
#rbfeps = 2/(rbf+1)


##### training parameters #####
max_epoch = 200
#start_sample = 99 #start evaluating on the test set at this epoch
#sampling_freq = 100 #evaluate on test set every sampling_freq number of epochs after you start sampling
start_sample = 0 #start evaluating on the test set at this epoch
sampling_freq = 1 #evaluate on test set every sampling_freq number of epochs after you start sampling
epoch_samples = [0, 24, 49, 74, 99, 124, 149, 174, 199]
test_size =  data_len // 10
bs = {'DHFR': 31, 'MUTAG': 17, 'COX2': 36, 'IMDB-BINARY': 90, 'NCI1': 137}
batch_size = bs[dataset_name]

train_batches = np.ceil((data_len-test_size)/batch_size).astype(int)
print('number of batches = ', train_batches, ' batch size = ', batch_size, ' test size ', test_size)

torch.manual_seed(0)
rng_state= torch.get_rng_state() #seed init to ensure same initial conditions for each training
#eval_model = ModelStatsRBF(rbf) # evaluation model
#coeffs = eval_model.rbfweights.detach()
#pickle.dump(coeffs.numpy(), open(result_dump + 'rbf_init.pkl' , 'wb'))
#ema_decay = 0.9

###### preprocess #####
data = []
label = []

for i in range(len(graph_list)):
    G = graph_list[i]
    L = nx.normalized_laplacian_matrix(G)
    w, v = eigh(L.todense())
    datum = dict()
    label.append(G.graph['label'])
    evals = torch.reshape(torch.from_numpy(w).float(), [len(G), 1])
    evecssq = torch.from_numpy(v**2).float()
    gram = 1/torch.sqrt((evals - centroids)**2/rbfeps**2 + 1)
    A = torch.matmul(evecssq, gram)
    #u = torch.matmul(A.T, torch.ones([len(w), 1])).flatten()
    #u = u/torch.norm(u)
    #P  = torch.matmul(A, u).unsqueeze(-1)*u #projection
    #datum['secondary_gram'] = A - P
    #datum['secondary_gram'] = A
    hks = np.matmul(evecssq, np.exp(-10*w)).flatten()
    if i ==0:
        Alist = A
        hkslist = hks
    else:
        Alist = np.append(Alist, A, axis = 0)
        hkslist = np.append(hkslist, hks)
    data.append(datum)
print(A.shape)
u, s, vh  = np.linalg.svd(Alist, full_matrices = False)
del Alist
gc.collect()
print(s)
#print(Alist.shape)
#print(s, u.shape)

#evals = torch.reshape(torch.linspace(0, 2, 200), [200, 1])
#gram = 1/torch.sqrt((evals - centroids)**2/rbfeps**2 + 1)
#gram = np.matmul(gram, vh.T )

#print(np.sum(np.abs(u), axis = 1))
#principal_dims = abs(s) >1e-2
#principal_dims = usum >100
#indices =np.where(principal_dims)
#sigma_inverse = np.abs(np.diag(1/s[principal_dims]))
#pds = len(sigma_inverse)
#print(pds)
#principals = np.matmul(vh[principal_dims].T, sigma_inverse)
#pickle.dump(principals, open(result_dump + 'principal_subspace.pkl', 'wb'))
#principals = torch.from_numpy(principals).float()
indices = [0,1,2]
print(s, indices)

winit = np.matmul(u[:,indices].T, hkslist)
reconstruct_delta = np.abs(np.matmul(u[:, indices], winit)- hkslist)
error_max = np.argmax(reconstruct_delta)
print(winit, winit.shape, reconstruct_delta[error_max], hkslist[error_max])

winit = torch.tensor(winit).float()
coeffs = winit
torch.set_rng_state(rng_state)
torch.manual_seed(0)


#eval_model = ModelStatsRBF(rbf = pds)
#coeffs = eval_model.rbfweights.detach().data.numpy()

mx, mn = -np.inf, np.inf
cnter = 0
for i in range(len(graph_list)):
    dat = data[i]
    G = graph_list[i]
    #dat['secondary_gram'] = torch.matmul(dat['secondary_gram'], principals)
    dat['secondary_gram'] = torch.from_numpy(u[cnter:cnter + len(G),indices])
    cnter += len(G)
    #hks = torch.flatten(torch.matmul(dat['secondary_gram'], torch.from_numpy(coeffs))).float().detach()
    hks = torch.flatten(torch.matmul(dat['secondary_gram'],winit)).float().detach()

    mx = max(float(torch.max(hks)), mx)
    mn = min(float(torch.min(hks)), mn)

    st = simplex_tree_constructor([list(e) for e in G.edges()])
    dat['simplex_tree'] = filtration_update(st, hks.numpy())
    dat['vectors'] = torch.zeros([1,18])
    #recompute persistence
    pers = dat['simplex_tree'].persistence(homology_coeff_field = 2)
    del pers

label = torch.tensor(label).float()
print(mn,  mx)
torch.set_rng_state(rng_state) #fix init state

#if fix wavelet
eval_model =  ModelStatsRBF(pds, mn = mn, mx=  mx, weightinit = winit)
eval_model.update = True
outcrap = eval_model(data)


print('Finished initial processing')
del graph_list
del u, s, vh, hkslist
gc.collect()
print(mn,  mx)
torch.set_rng_state(rng_state)
####### torch random seeds #######
shuffidx = list(range(data_len)) # data indexer

criterion = nn.BCEWithLogitsLoss() #loss function

for run in range(10):
    print('run = ', run)
    np.random.seed(run)
    np.random.shuffle(shuffidx) # randomly choose partition of data into test / fold

    for fold in range(10):
        print ('> fold ', fold)
        test_acc = []
        train_acc = []
        loss_func = []
        rbf_params = []

        test_bottom = fold * test_size
        test_top = (1+fold) * test_size
        test_indices = shuffidx[test_bottom : test_top]
        train_indices = shuffidx[0:test_bottom] + shuffidx[test_top :]

        torch.set_rng_state(rng_state) #fix init state
        pht = ModelStatsRBF(pds, mn = mn, mx=  mx, weightinit = winit)#### Mode
        criterion = nn.BCEWithLogitsLoss() #loss function

        #coeffs = pht.rbfweights.detach().data
        optimizer_classifier = optim.Adam([{'params': [param for name, param in pht.named_parameters() if 'rbfweights' not in name]}], lr=1e-3, weight_decay = 0.0)
        optimizer_filter = optim.SGD([{'params': [param for name, param in pht.named_parameters() if 'rbfweights' in name]}], lr=1e-2)

        for epoch in range(max_epoch):
            pht.train()
            np.random.shuffle(train_indices)

            if epoch == 200:
                pht.update = True
                outputs = pht(data) # update all frozen vectors to latest parameters
                pht.freeze_persistence = True

                for name, param in pht.named_parameters():
                    if 'rbfweights' in name:
                        param.requires_grad = False


            tna = 0
            lss = 0

            for b in range(train_batches):

                train_indices_batch = train_indices[b*batch_size : (b+1)*batch_size ]
                mod_labels = (label[train_indices_batch] + torch.tensor(np.random.binomial(1, relabel_p, len(train_indices_batch))))%2
                mod_labels = mod_labels*(1-2*smooth_squeeze) + smooth_squeeze

                optimizer_classifier.zero_grad()
                optimizer_filter.zero_grad()

                #outputs, regulariser = pht([data[i] for i in train_indices_batch])
                outputs = pht([data[i] for i in train_indices_batch])
                #loss = criterion(outputs, label[train_indices_batch].unsqueeze(-1)) + regulariser_constant*regulariser
                loss = criterion(outputs, mod_labels.unsqueeze(-1))
                #loss = criterion(outputs, label[train_indices_batch].unsqueeze(-1))
                loss.backward()
                #nn.utils.clip_grad_norm_(pht.rbfweights.parameters(), 0.5)
                #nn.utils.clip_grad_norm_(pht.rbfweights, 0.5)
                #optimizer_classifier.step()
                #optimizer_filter.step()
                if epoch < 200:
                    optimizer_classifier.step()
                    optimizer_filter.step()
                else:
                    optimizer_classifier.step()

                tna += int(((outputs.view(-1) > 0) == label[train_indices_batch]).sum())
                lss += float(loss)*len(train_indices_batch)

                #train_acc.append(int(((outputs.view(-1) > 0) == label[train_indices_batch]).sum())/len(train_indices_batch))
                #loss_func.append(float(loss))
                #dtheta.append(float(pht.rbfweights.grad.data.norm(2)))
                #theta.append(pht.rbfweights.detach().clone().numpy())

            if epoch < 200:
                rbf_params.append(pht.rbfweights.detach().clone().numpy())
            train_acc.append(tna/len(train_indices))
            loss_func.append(lss/len(train_indices))

            if epoch  in epoch_samples:
                #tp = 0
                #for p in pht.rbfweights.parameters():
                #    tp += p.grad.data.norm(2)**2
                pht.eval()
                test_outputs = pht([data[i] for i in test_indices])

                #train_outputs =pht([data[i] for i in train_indices])

                test_acc.append(int(((test_outputs.view(-1) > 0) == label[test_indices]).sum())/len(test_indices))
                #loss_func.append(criterion(train_outputs, label[train_indices].unsqueeze(-1)).item())
                #train_acc.append(int(((train_outputs.view(-1) > 0) == label[train_indices]).sum())/len(train_indices))
                #if (epoch - start_sample) % (2*sampling_freq) == 0:
                #    print(loss_func[-1], testn_acc[-1], train_acc[-1], float(pht.rbfweights.grad.norm(2)))


        #pht.eval()
        # saves the final learnt filter #
        run_fold_index = str(run) + '_' + str(fold)
        saverunfoldfilter = result_dump + 'learnt_filter_' + run_fold_index
        #pickle.dump(eval_model.ev_function(torch.linspace(0,2, 100).unsqueeze(1)).squeeze(1).detach().numpy(), open(saverunfoldfilter , 'wb')) #sampled from lambda = 0... 2 at 100 evenly spaced points
        #pickle.dump(pht.rbfweights.detach().numpy(), open(saverunfoldfilter , 'wb')) #sampled from lambda = 0... 2 at 100 evenly spaced points

        #pickle.dump(train_acc, open(result_dump + 'train_acc_' + run_fold_index, 'wb'))
        #pickle.dump(test_acc, open(result_dump + 'test_acc_' + run_fold_index, 'wb'))
        #pickle.dump(loss_func, open(result_dump + 'loss_' + run_fold_index, 'wb'))
        #pickle.dump(dtheta, open(result_dump + 'dtheta_' + run_fold_index, 'wb'))
        #pickle.dump(theta, open(result_dump + 'theta_' + run_fold_index, 'wb'))

        pickle.dump(train_acc, open(result_dump + 'train_acc_'+ run_fold_index + '.pkl', 'wb'))
        pickle.dump(test_acc, open(result_dump + 'test_acc_' + run_fold_index + '.pkl', 'wb'))
        pickle.dump(loss_func, open(result_dump + 'loss_' + run_fold_index+'.pkl', 'wb'))
        pickle.dump(rbf_params, open(result_dump + 'theta_' + run_fold_index+ '.pkl', 'wb'))


        del rbf_params, train_acc, test_acc, loss_func, pht
        gc.collect()
