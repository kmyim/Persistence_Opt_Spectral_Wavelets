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
from models import ModelPIRBFDoubleOneStatic, EMA, GenPHandPI
import matplotlib.pyplot as plt
import iisignature
from collections import Counter


##### neural net parameters #####
gc.collect()
#torch.set_num_threads(4)
# persistence image
resolution = 20 # res x res image
expt_name = 'rbf12_bugfix2_nofeats_static'
param = 'rbf'
bfs = 12
new_bfs = 12
s_cutoff = 0

lr = 1e-2
lr_decay = False
lr_cl = 1e-3
ema_decay = 0.9
xtra_feat = False  #if  true, add extra features to model
xxtra = False
Takens = True
switch  = False
random_init = False
class_weighting = False
weight_decay = 0
max_epoch = 200
wavelet_opt = 0
epoch_samples = [k for k in [0, 24, 49, 75, 99, 124, 149, 174, 199, 224, 249] if k < max_epoch]
#epoch_samples = [0, 9, 19, 29, 39, 49]



if param == 'rbf':
    bfseps = 2/(bfs-3)
    centroids = torch.linspace(-bfseps, 2 + bfseps, bfs)


##### directories #####
dataset_name = 'MUTAG/'
raw = 'raw_datasets/'
processed = 'data_example/'
result_dump = 'ten10folds/' + dataset_name + expt_name + '/' + expt_name + '_'


##### load data #####

graph_list = pickle.load(open(processed + dataset_name + 'networkx_graphs.pkl', 'rb'))
data_len = len(graph_list)

##### training parameters #####
bs = {'DHFR/': 11, 'MUTAG/': 10, 'COX2/': 9, 'IMDB-BINARY/': 18, 'NCI1/': 20, 'IMDB-MULTI/': 27 }
#bs = {'DHFR/': 59, 'MUTAG/': 57, 'COX2/': 47, 'IMDB-BINARY/': 50, 'NCI1/': 20, 'IMDB-MULTI/': 27, 'PROTEINS/': 59 }

batch_size = bs[dataset_name]
test_size =  data_len // 10
train_batches = np.ceil((data_len-test_size)/batch_size).astype(int)
print('number of batches = ', train_batches, ' batch size = ', batch_size)

torch.manual_seed(0)
rng_state= torch.get_rng_state() #seed init to ensure same initial conditions for each training

### eigenvalue path signature ####
if xtra_feat:
    pslevel = 4
    sig_prep = iisignature.prepare(2, pslevel)
    #xtra_feat_length = iisignature.logsiglength(2, pslevel)
    siglength = iisignature.logsiglength(2, pslevel)

    xtra_feat_length = siglength
    if xxtra: xtra_feat_length += 4

else:
    xtra_feat_length = 0



###### preprocess #####
data = []
label = []
mx, mn = -np.inf, np.inf
Alist = None
Aker = None
smax = []
for i in range(len(graph_list)):
    G = graph_list[i]
    L = nx.normalized_laplacian_matrix(G)
    lam, v = eigh(L.todense())
    w = torch.from_numpy(lam).float()

    if len(G)*(len(G)-1) > 2*G.number_of_edges():
        diameter = 2
    else:
        diameter = 0

    datum = dict()
    label.append(G.graph['label'] % 2)

    if xtra_feat:
        if Takens:
            path =  np.zeros([len(lam)-1, 2]) #taken's embedding of eigenvalues
            path[:,0] = lam[1:]
            path[:,1] = lam[:-1]
        else:
            path =  np.zeros([len(lam), 2])
            path[:,0] = lam
            path[:,1] = np.linspace(0, 2, len(lam))
        #datum['feats'] = torch.tensor(iisignature.logsig(path,sig_prep)).float()

        sigs = torch.tensor(iisignature.logsig(path,sig_prep)).float()
        if xxtra:
            feats = torch.zeros([xtra_feat_length])
            feats[4:] = sigs
        else:
            datum['feats'] = sigs

    evecssq = torch.from_numpy(v**2).float()
    if param == 'rbf':
        gram = 1/torch.sqrt((torch.reshape(w, [len(G), 1]) - centroids)**2/bfseps**2 + 1)
        A = torch.matmul(evecssq, gram)
        #A = A - torch.mean(A, dim = 0)

    elif param == 'cheb':
        vander=  torch.from_numpy(np.polynomial.chebyshev.chebvander(lam, bfs+1)).float()
        A = torch.matmul(evecssq, vander[:, 2:])


    hks = torch.matmul(evecssq, torch.exp(-10*w)).flatten()
    hks_another = torch.matmul(evecssq, torch.exp(w)).flatten()

    #hks = hks - torch.mean(hks)

    datum['f'] = hks
    datum['f_static'] =  hks_another
    if xxtra and xtra_feat:
        feats[0] = torch.min(hks_another)
        feats[1] = torch.max(hks_another)
        feats[2] = torch.min(hks)
        feats[3] = torch.max(hks)
        datum['feats'] = feats
    #feats[4] = diameter

    datum['diameter'] = diameter

    #datum['fminmax'] = torch.zeros(2)
    st = simplex_tree_constructor([list(e) for e in G.edges()])
    datum['simplex_tree'] = filtration_update(st, hks.numpy())
    datum['images'] = torch.zeros([6, resolution, resolution])
    #recompute persistence
    pers = datum['simplex_tree'].persistence(homology_coeff_field = 2)

    if diameter > 1:
        mx = max(float(torch.max(hks_another)), mx)
        mn = min(float(torch.min(hks_another)), mn)
        #u_g, s_g, vh_g  = np.linalg.svd(A, full_matrices = False)
        if type(Alist) == type(None):
            Alist = A
            hkslist = hks
        else:
            Alist = np.append(Alist, A, axis = 0)
            hkslist = np.append(hkslist, hks)

        #if type(Aker) == type(None):
        #    ker = vh_g[s_g<s_cutoff*max(s_g), :]
        #    Aker = vh_g[s_g > s_cutoff, :]
        #else:
            #Aker = np.append(Aker, vh_g[s_g<s_cutoff*max(s_g), :], axis = 0 )
            #Aker = np.append(Aker, vh_g[s_g > s_cutoff, :], axis = 0 )

    data.append(datum)
print(Counter(label))

#u_ker, s_ker, vh_ker  = np.linalg.svd(Aker, full_matrices = False)

#plt.plot(s_ker, '.-')
#plt.show()

#t = torch.linspace(0, 2, 50)
#if param == 'rbf':
#    gram = 1/torch.sqrt((torch.reshape(t, [-1, 1]) - centroids)**2/bfseps**2 + 1)
#    for i in range(4):
#        plt.plot(t, torch.matmul(gram, torch.reshape(torch.from_numpy(vh_ker[i]), [-1, 1]) ), label = str(i))
#    plt.legend()
#    plt.xlabel('eigenvalues')
#    plt.savefig('imdb_b_rbf102_shared_wavelet.pdf', dpi = 300)
#elif param == 'cheb':
#    vander=  torch.from_numpy(np.polynomial.chebyshev.chebvander(t, bfs+1)).float()[:, 2:]
    #for i in range(new_bfs):
        #plt.plot(t, torch.matmul(vander, torch.reshape(torch.from_numpy(vh_ker[i]), [-1, 1]) ))
    #plt.show()

#del Aker

genphandpi = GenPHandPI(resolution, lims = [mn, mx], filtername = 'f_static' )
PIs_static = genphandpi(data)


for i in range(len(graph_list)):
    dat = data[i]
    dat['images'][3:, :,:] = PIs_static[i]

#Alist = np.matmul(Alist, vh_ker[0:new_bfs,:].T)
u, s, vh  = np.linalg.svd(Alist, full_matrices = False)
del Alist
gc.collect()
print('u shape, ', u.shape)
print('s ', s)

u = u[:, s > s_cutoff]
vh = vh[s > s_cutoff, :]
new_bfs = sum(s> s_cutoff)

torch.set_rng_state(rng_state)
torch.manual_seed(0)


if random_init:
    winit = torch.empty(new_bfs, 1).normal_(mean = 0, std = 1/np.sqrt(new_bfs))
else:
    winit = np.matmul(u.T, hkslist)
    reconstruct_delta = np.abs(np.matmul(u, winit)- hkslist)
    error_max = np.argmax(reconstruct_delta)
    #print(winit, winit.shape)
    print('max error', reconstruct_delta[error_max], hkslist[error_max])
    winit = torch.tensor(winit).float()

mx, mn = -np.inf, np.inf
cnter = 0
for i in range(len(graph_list)):
    dat = data[i]
    if dat['diameter'] > 1:
        G = graph_list[i]
        dat['secondary_gram'] = torch.from_numpy(u[cnter:cnter + len(G),:])
        cnter+= len(G)
        hks = torch.flatten(torch.matmul(dat['secondary_gram'], winit)).float().detach()

        mx = max(float(torch.max(hks)), mx)
        mn = min(float(torch.min(hks)), mn)

print('min, max', mn,  mx)
del u, s, vh, PIs_static

gc.collect()


#if fix wavelet
#eval_model = ModelPIRBFDoubleOneStatic(bfs = bfs, resolution = resolution, lims = [mn, mx], weightinit = winit, extra_feat_len = xtra_feat_length)
#eval_model.update = True #write PIs to dat['images']
#outcrap = eval_model(data)

label = torch.tensor(label).float()
print('Finished initial processing')
del graph_list
gc.collect()

torch.set_rng_state(rng_state) #fix init state
shuffidx = list(range(data_len)) # data indexer
criterion = nn.BCEWithLogitsLoss() #loss function


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
        numones = sum(label[train_indices])
        pos_weight = (len(train_indices) - numones)/numones

        train_acc = []
        loss_func = []
        bfs_params = []
        bfs_grad = []
        test_acc = []

        torch.set_rng_state(rng_state) #fix init state
        torch.manual_seed(0)
        pht = ModelPIRBFDoubleOneStatic(rbf = new_bfs, resolution = resolution, lims = [mn, mx], weightinit = winit, extra_feat_len = xtra_feat_length)

        #pht = ModelPIRBFDoubleOneStatic(bfs = bfs, resolution = resolution, lims = [mn, mx])
        #if run==0 and fold == 0: print(pht.rbfweights)


        torch.set_rng_state(rng_state)
        torch.manual_seed(0)

        eval_model = ModelPIRBFDoubleOneStatic(rbf = new_bfs, resolution = resolution, lims = [mn, mx], weightinit = winit, extra_feat_len = xtra_feat_length)

        eval_model.update = True
        outcrap = eval_model(data)
        del outcrap
        gc.collect()
        #for i in range(6):
        #    plt.imshow(data[-1]['images'][i])
        #    plt.show()

        ##### If fix wavelet #######
        #if wavelet_opt <= 0:
        #    pht.freeze_persistence = True
        #    eval_model.freeze_persistence = True
        #    for name, param in pht.named_parameters():
        #        if 'rbfweights' in name:
        #            param.requires_grad = False
        #        else:
        #            param.requires_grad = True


        ema = EMA(ema_decay)
        ema.register(pht.state_dict(keep_vars = False))

        if class_weighting:
            criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight) #loss function
        else:
            criterion = nn.BCEWithLogitsLoss() #loss function

        optimizer_classifier = optim.Adam([{'params': [param for name, param in pht.named_parameters() if 'rbfweights' not in name]}], lr=lr_cl, weight_decay = 0)
        optimizer_filter = optim.SGD([{'params': pht.rbfweights}], lr=lr, weight_decay = weight_decay)

        if lr_decay :
            lambda1 = lambda epoch: 0.95 ** epoch
            scheduler = optim.lr_scheduler.LambdaLR(optimizer_filter, lr_lambda= lambda1)

        for epoch in range(max_epoch):
            pht.train()
            np.random.shuffle(train_indices)

            if epoch == wavelet_opt:


                pht.freeze_persistence = False
                pht.update = True
                outputs = pht(data) #with update  = true, computes and stores all images
                pht.freeze_persistence = True

                for name, param in pht.named_parameters():
                    if 'rbfweights' in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

                eval_model.freeze_persistence = True
            tna = 0
            lss = 0

            for b in range(train_batches):

                train_indices_batch = train_indices[b*batch_size : (b+1)*batch_size ]
                mod_labels = label[train_indices_batch]
                optimizer_classifier.zero_grad()
                optimizer_filter.zero_grad()

                outputs = pht([data[i] for i in train_indices_batch])

                loss = criterion(outputs, mod_labels.unsqueeze(-1))
                loss.backward()

                if epoch < wavelet_opt:
                    optimizer_filter.step()
                    if not switch: optimizer_classifier.step()
                else:
                    optimizer_classifier.step()


                pht.eval()
                ema.update(pht.state_dict(keep_vars = False))
                pht.train()

                tna += int(((outputs.view(-1) > 0) == label[train_indices_batch]).sum())
                lss += float(loss)
                #train_acc.append(int(((outputs.view(-1) > 0) == label[train_indices_batch]).sum())/len(train_indices_batch))
                #loss_func.append(float(loss))


            if lr_decay: scheduler.step()
            if epoch < wavelet_opt:
                bfs_params.append(pht.rbfweights.detach().clone().numpy())

            train_acc.append(tna/len(train_indices))
            loss_func.append(lss/len(train_indices))

            if epoch in epoch_samples:
                eval_model.load_state_dict(ema.shadow)
                eval_model.eval()
                pht.eval()
                #test_outputs = pht([data[i] for i in test_indices])

                test_outputs = eval_model([data[i] for i in test_indices])
                test_acc.append(int(((test_outputs.view(-1) > 0) == label[test_indices]).sum())/len(test_indices))



        run_fold_index = str(run) + '_' + str(fold)
        pickle.dump(train_acc, open(result_dump + 'train_acc_'+ run_fold_index + '.pkl', 'wb'))
        pickle.dump(test_acc, open(result_dump + 'test_acc_' + run_fold_index + '.pkl', 'wb'))
        pickle.dump(loss_func, open(result_dump + 'loss_' + run_fold_index+'.pkl', 'wb'))
        #pickle.dump(bfs_grad, open(result_dump + 'dtheta_'+ run_fold_index+'.pkl', 'wb'))
        pickle.dump(bfs_params, open(result_dump + 'theta_' + run_fold_index+ '.pkl', 'wb'))
        #print(max(train_acc), train_acc[-1], max(test_acc), test_acc[-1])

        del bfs_params, train_acc, test_acc, loss_func, eval_model, pht

        if lr_decay:
            del scheduler

        gc.collect()
