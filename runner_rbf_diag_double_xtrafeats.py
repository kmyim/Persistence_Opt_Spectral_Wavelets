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


##### neural net parameters #####
gc.collect()
# persistence image
resolution = 20 # res x res image
error_tol = 0 #only recompute persistence when change in filter function > error_tol
expt_name = 'rbf12_diag_double_picnn_eigsig_bugfix'
rbf = 12
lr = 1e-2
ema_decay = 0.9
xtra_feat = True #if  true, add extra features to model

rbfeps = 2/(rbf-3)
centroids = torch.linspace(-rbfeps, 2 + rbfeps, rbf)


##### directories #####
dataset_name = 'COX2/'
raw = 'raw_datasets/'
processed = 'data_example/'
result_dump = 'ten10folds/' + dataset_name + expt_name + '/' + expt_name + '_'


##### load data #####

graph_list = pickle.load(open(processed + dataset_name + 'networkx_graphs.pkl', 'rb'))
data_len = len(graph_list)

##### training parameters #####
max_epoch =200
wavelet_opt = 50
epoch_samples = [0, 24, 49, 75, 99, 124, 149, 174, 199]
bs = {'DHFR/': 11, 'MUTAG/': 10, 'COX2/': 9, 'IMDB-BINARY/': 18, 'NCI1/': 20, 'IMDB-MULTI/': 27 }
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
    xtra_feat_length = iisignature.logsiglength(2, pslevel)
else:
    xtra_feat_length = 0

print(xtra_feat_length)



###### preprocess #####
data = []
label = []
mx, mn = -np.inf, np.inf

for i in range(len(graph_list)):
    G = graph_list[i]
    L = nx.normalized_laplacian_matrix(G)
    lam, v = eigh(L.todense())
    w = torch.from_numpy(lam).float()

    datum = dict()
    label.append(G.graph['label'])

    if xtra_feat:
        path =  np.zeros([len(lam)-1, 2]) #taken's embedding of eigenvalues
        path[:,0] = lam[1:]
        path[:,1] = lam[:-1]
        datum['feats'] = torch.tensor(iisignature.logsig(path,sig_prep)).float()


    evecssq = torch.from_numpy(v**2).float()
    gram = 1/torch.sqrt((torch.reshape(w, [len(G), 1]) - centroids)**2/rbfeps**2 + 1)
    A = torch.matmul(evecssq, gram)

    hks = torch.matmul(evecssq, torch.exp(-10*w)).flatten()
    hks_another = torch.matmul(evecssq, torch.exp(w)).flatten()

    datum['f'] = hks
    datum['f_static'] =  hks_another


    st = simplex_tree_constructor([list(e) for e in G.edges()])
    datum['simplex_tree'] = filtration_update(st, hks.numpy())
    datum['images'] = torch.zeros([6, resolution, resolution])
    #recompute persistence
    pers = datum['simplex_tree'].persistence(homology_coeff_field = 2)


    mx = max(float(torch.max(hks_another)), mx)
    mn = min(float(torch.min(hks_another)), mn)

    if i ==0:
        Alist = A
        hkslist = hks
    else:
        Alist = np.append(Alist, A, axis = 0)
        hkslist = np.append(hkslist, hks)
    data.append(datum)


genphandpi = GenPHandPI(resolution, lims = [mn, mx], filtername = 'f_static' )
PIs_static = genphandpi(data)


for i in range(len(graph_list)):
    dat = data[i]
    dat['images'][3:, :,:] = PIs_static[i]

#plt.hist(hks)
#plt.show()
u, s, vh  = np.linalg.svd(Alist, full_matrices = False)
del Alist
gc.collect()

winit = np.matmul(u.T, hkslist)
reconstruct_delta = np.abs(np.matmul(u, winit)- hkslist)
error_max = np.argmax(reconstruct_delta)
print(winit, winit.shape)
print(reconstruct_delta[error_max], hkslist[error_max])

winit = torch.tensor(winit).float()
coeffs = winit
torch.set_rng_state(rng_state)
torch.manual_seed(0)


mx, mn = -np.inf, np.inf
cnter = 0
for i in range(len(graph_list)):
    dat = data[i]
    G = graph_list[i]
    dat['secondary_gram'] = torch.from_numpy(u[cnter:cnter + len(G),:])
    cnter+= len(G)
    hks = torch.flatten(torch.matmul(dat['secondary_gram'], winit)).float().detach()
    mx = max(float(torch.max(hks)), mx)
    mn = min(float(torch.min(hks)), mn)

#print(mn,mx)



del u, s, vh, PIs_static

gc.collect()


#if fix wavelet
eval_model = ModelPIRBFDoubleOneStatic(rbf = rbf, resolution = resolution, lims = [mn, mx], weightinit = winit, extra_feat_len = xtra_feat_length)
eval_model.update = True #write PIs to dat['images']
outcrap = eval_model(data)

label = torch.tensor(label).float()
print('Finished initial processing')
del graph_list, outcrap
gc.collect()
print(mn,  mx)
torch.set_rng_state(rng_state) #fix init state
shuffidx = list(range(data_len)) # data indexer
criterion = nn.BCEWithLogitsLoss() #loss function


for run in range(1, 10):
    print('run = ', run)
    np.random.seed(run)
    np.random.shuffle(shuffidx) # randomly choose partition of data into test / fold

    for fold in range(10):
        print ('> fold ', fold)

        test_bottom = fold * test_size
        test_top = (1+fold) * test_size
        test_indices = shuffidx[test_bottom : test_top]
        train_indices = shuffidx[0:test_bottom] + shuffidx[test_top :]


        train_acc = []
        loss_func = []
        rbf_params = []
        rbf_grad = []
        test_acc = []

        torch.set_rng_state(rng_state) #fix init state
        torch.manual_seed(0)
        pht = ModelPIRBFDoubleOneStatic(rbf = rbf, resolution = resolution, lims = [mn, mx], weightinit = winit, extra_feat_len = xtra_feat_length)
        #pht = ModelPIRBFDoubleOneStatic(rbf = rbf, resolution = resolution, lims = [mn, mx])
        #if run==0 and fold == 0: print(pht.rbfweights)


        torch.set_rng_state(rng_state)
        torch.manual_seed(0)
        eval_model = ModelPIRBFDoubleOneStatic(rbf = rbf, resolution = resolution, lims = [mn, mx], weightinit = winit, extra_feat_len = xtra_feat_length)
        #eval_model.freeze_persistence = True
        #if run==0 and fold == 0: print(eval_model.rbfweights)

        ##### If fix wavelet #######
        if wavelet_opt <= 0:
            pht.freeze_persistence = True
            for name, param in pht.named_parameters():
                if 'rbfweights' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True


        ema = EMA(ema_decay)
        ema.register(pht.state_dict(keep_vars = False))

        criterion = nn.BCEWithLogitsLoss() #loss function

        optimizer_classifier = optim.Adam([{'params': [param for name, param in pht.named_parameters() if 'rbfweights' not in name]}], lr=1e-3, weight_decay = 0.0)
        optimizer_filter = optim.SGD([{'params': pht.rbfweights}], lr=lr)

        for epoch in range(max_epoch):
            pht.train()
            np.random.shuffle(train_indices)
            #if epoch == 1:
            #    for name, param in pht.named_parameters():
            #        if 'CNN' in name:
            #            param.requires_grad = False

            if epoch == wavelet_opt:
                #eval_model.update = True
                #outputs = eval_model(data) # update all frozen vectors to latest parameters
                eval_model.freeze_persistence = True
                pht.update = True
                outputs = pht(data)
                pht.freeze_persistence = True

                for name, param in pht.named_parameters():
                    if 'CNN' in name:
                        param.requires_grad = True
                    if 'rbfweights' in name:
                        param.requires_grad = False
                #if run == 0 and epoch == 0:
                #    fig, ((ax1, ax2, ax7), (ax3, ax4, ax8) , (ax5, ax6, ax9))  = plt.subplots(3,3)
                #    ax1.imshow(data[-47]['images'][0,0])
                #    #ax2.imshow(placeholder[0])
                #    ax3.imshow(data[-47]['images'][0,1])
                #    #ax4.imshow(placeholder[1])
                #    ax5.imshow(data[-47]['images'][0,2])
                    #ax6.imshow(placeholder[2])
                    #ax7.imshow(placeholder[0] - data[0]['images'][0,0])
                    #ax8.imshow(placeholder[1] - data[0]['images'][0,1])
                    #ax9.imshow(placeholder[2] - data[0]['images'][0,2])
                #    plt.show()
                    #crap
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
                    optimizer_classifier.step()
                    optimizer_filter.step()
                else:
                    optimizer_classifier.step()


                pht.eval()
                ema.update(pht.state_dict(keep_vars = False))
                pht.train()

                tna += int(((outputs.view(-1) > 0) == label[train_indices_batch]).sum())
                lss += float(loss)
                #train_acc.append(int(((outputs.view(-1) > 0) == label[train_indices_batch]).sum())/len(train_indices_batch))
                #loss_func.append(float(loss))
                #rbf_grad.append(float(pht.rbfweights.grad.data.norm(2)))
            eval_model.load_state_dict(ema.shadow)
            eval_model.eval()
            if epoch < wavelet_opt:
                rbf_params.append(eval_model.rbfweights.detach().clone().numpy())

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
        #pickle.dump(rbf_grad, open(result_dump + 'dtheta_'+ run_fold_index+'.pkl', 'wb'))
        pickle.dump(rbf_params, open(result_dump + 'theta_' + run_fold_index+ '.pkl', 'wb'))
        print(max(train_acc), train_acc[-1], max(test_acc), test_acc[-1])

        del rbf_params, train_acc, test_acc, loss_func, eval_model, pht
        gc.collect()
