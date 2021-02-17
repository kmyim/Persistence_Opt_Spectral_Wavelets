import numpy as np
import gudhi
import torch


def simplex_tree_constructor0dim(G):
    '''
    G a graph, list of edges [v,w]
    output gudhi simplex tree object
    '''

    st = gudhi.SimplexTree()
    st.insert([1000], filtration = -1000)

    st.persistence()
    for e in G:
        st.insert(e, 1.0)
    st.initialize_filtration()
    st.persistence(homology_coeff_field = 2)
    bs = st.betti_numbers()
    if len(bs) >= 2:
        b1 = bs[1]
    else:
        b1 = 0

    for e in G:
        st.insert([1000] + e, 1.0) #cone
    st.initialize_filtration()
    return st, b1

def simplex_tree_constructor(G, flag=False):
    '''
    G a graph, list of edges [v,w]
    output gudhi simplex tree object
    '''

    st = gudhi.SimplexTree()
    st.insert([1000], filtration = -1000)

    for e in G:
        st.insert(e, 0.0)
        st.insert([1000] + e, 1000) #cone

    if flag:
        st.persistence()
        betti1 = st.betti_numbers()[1]
        st.expansion(2)
        st.persistence()
        new_betti1 = st.betti_numbers()[1]
        a = [betti1, new_betti1]
        return st, a

    else:
        return st

def simplex_tree_constructor_ord(G):
    '''
    G a graph, list of edges [v,w]
    output gudhi simplex tree object
    '''

    st = gudhi.SimplexTree()

    for e in G:
        st.insert(e, 0.0)
    return st

def filtration_update(st, f):

    fmax = max(f)
    fmin = min(f)
    st.assign_filtration(simplex = [1000], filtration = -1000)

    for s in st.get_skeleton(1):
        if len(s[0]) == 1 and 1000 not in s[0] :
            st.assign_filtration(simplex = s[0], filtration = f[s[0]])
        if len(s[0]) == 2:
            if 1000 in s[0]:
                st.assign_filtration(simplex = s[0], filtration = 1000 - min(f[v] for v in s[0] if v != 1000))
            else:
                st.assign_filtration(simplex = s[0], filtration = max(f[v] for v in s[0]))

    #for v in range(len(f)):
    #    st.assign_filtration(simplex = [v], filtration = f[v])

    #for s in st.get_cofaces([1000], codimension = 1):
    #    if len(s[0]) == 2: st.assign_filtration(simplex = s[0], filtration = 1000 - min(f[v] for v in s[0] if v != 1000))

    for s in st.get_cofaces([1000], codimension = 2):
        if len(s[0]) == 3: st.assign_filtration(simplex = s[0], filtration = 1000 - min(f[v] for v in s[0] if v != 1000))

    st.initialize_filtration()
    st.make_filtration_non_decreasing()

    return st

def filtration_update_ord(st, f):

    for s in st.get_skeleton(1):
        if len(s[0]) == 1:
            st.assign_filtration(simplex = s[0], filtration = f[s[0]])
        if len(s[0]) == 2:
            st.assign_filtration(simplex = s[0], filtration = max(f[v] for v in s[0]))

    st.initialize_filtration()
    st.make_filtration_non_decreasing()

    return st
