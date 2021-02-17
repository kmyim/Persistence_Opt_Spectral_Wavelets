import networkx as nx
import torch
import pickle
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from playground import *
import os

dataset_name = 'PROTEINS'
raw = 'data_example/raw/'
processed = 'data_example/'

edge_list = raw + dataset_name + '/' + dataset_name + '_A.txt'
graph_index = raw + dataset_name + '/' + dataset_name + '_graph_indicator.txt'
graph_label = raw + dataset_name + '/' + dataset_name + '_graph_labels.txt'

#edge_list = raw + dataset_name + '/' + 'edges.txt'
#graph_index = raw + dataset_name + '/' + 'graph_idx.txt'
#graph_label = raw + dataset_name + '/' + 'graph_labels.txt'

dump_dir = processed + dataset_name
result_dump = dump_dir + '/networkx_graphs.pkl'
os.makedirs(dump_dir)
if os.path.isdir(dump_dir) == False:
    print('Warning, no path at ' + dump_dir)

edge_file = open(edge_list, 'r')
index_file = open(graph_index, 'r')
label_file = open(graph_label, 'r')

all_edges = edge_file.readlines()
all_labels = label_file.readlines()
all_indices  = index_file.readlines()
L = len(all_edges)
counter = 1
graph_list = []

ticker = 0
G = nx.Graph(label= max(0, int(all_labels[counter-1].strip('\n '))))
for i in range(L):
        s = all_edges[i]
        e = [int(a)for a in s.strip('\n ').split(',')]
        glab = int(all_indices[e[0]-1 ].strip('\n'))
        if glab != counter:
                print(i, glab, e[0], counter)
                G = nx.relabel.convert_node_labels_to_integers(G)
                graph_list.append(G)
                ticker += len(G)
                G = nx.Graph(label= max(0,int(all_labels[counter].strip('\n '))))
                counter += 1
        G.add_edge(e[0] - ticker - 1, e[1] - ticker - 1)
G = nx.relabel.convert_node_labels_to_integers(G)
graph_list.append(G)

print(len(graph_list), len(all_labels))
nx.draw_kamada_kawai(G)
plt.show()

error_count = 0
for g in graph_list:
        if not nx.is_connected(g):
                error_count += 1
print(error_count)

pickle.dump(graph_list, open(result_dump, 'wb'))
