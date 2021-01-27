
class ModelChebyshevStats(nn.Module):

    def __init__(self, cheby_degree = 6):

        super(ModelChebyshevStats, self).__init__()

        self.cheby_params = torch.nn.Parameter(torch.empty([cheby_degree, 1]).normal_(mean = 0, std = 1/np.sqrt(cheby_degree*np.pi/2)), requires_grad = True)
        #self.cheby_params = torch.nn.Parameter(torch.tensor([ 0.12796638, -0.2425746 ,  0.20752221, -0.15955385,  0.11156322, -0.07059596,  0.04106911, -0.02148135,  0.01064266, -0.0046641 , 0.0020595 ]), requires_grad = True)
        self.max_num_intervals = [50 ,50, 50]
        self.features = DiagramStats()
        self.project = nn.Sequential(nn.BatchNorm1d(18, affine = False), nn.Linear(18,1))

    def forward(self, mb):

        L = len(mb)
        births = torch.zeros([L, 3, max(self.max_num_intervals)])
        deaths = torch.zeros([L, 3, max(self.max_num_intervals)])

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
        features = self.features(births, deaths)
        features = torch.reshape(features, [L, -1])
        del births, deaths

        return self.project(features)

class PiecewiseMonotonic(nn.Module):

    def __init__(self, planes, groups, sign = -1):

        super(PiecewiseMonotonic, self).__init__()

        #self.maxlayer = nn.MaxPool1d(groups,groups)
        self.planes = planes #planes per group
        self.groups = groups
        self.sign = sign
        #
        self.weights = nn.Parameter(torch.empty(1, planes*groups).uniform_(0, 2), requires_grad = True)
        #
        if sign <0 :
            self.bias = nn.Parameter(torch.empty(1, planes*groups).uniform_(4, 5), requires_grad = True)
        else:
            self.bias = nn.Parameter(torch.empty(1, planes*groups).uniform_(0, 1), requires_grad = True)
        #self.weights = nn.Parameter(torch.empty(1, planes*groups).uniform_(np.log(1/2), np.log(2)), requires_grad = True)
        #self.bias = nn.Parameter(torch.empty(1, planes*groups).uniform_(1, 2), requires_grad = True)
        #self.weights = torch.nn.Parameter(2*torch.rand([1, planes*groups]), requires_grad = True)
        #self.bias = torch.nn.Parameter(torch.rand([1, planes*groups])+ 1, requires_grad = True)
        self.softplus = nn.Softplus()

        self.ffmp = nn.Sequential(nn.Linear(1, planes), nn.ReLU(), nn.Linear(planes, groups), nn.ReLU(), nn.Linear(groups, 1), nn.BatchNorm1d(1, affine = False))

    def forward(self, x):
        #x = self.sign*x * torch.exp(self.weights)  + self.bias

        ##x = self.maxlayer(x.unsqueeze(1))
        ##x = torch.min(x, axis = 2).values

        #x= x.reshape([x.shape[0], self.groups, self.planes])
        #x = torch.logsumexp(x, dim = 2) #softmax over each group
        #x = -torch.logsumexp(-x, dim = 1)

        #return x
        return self.ffmp(x)

class ModelStats(nn.Module):

    def __init__(self, cheby_degree = 5, square = [0,1]):

        super(ModelStats, self).__init__()
        self.features = DiagramStats()
        self.project = nn.Linear(18, 1)
        #self.project = feed_forward_mlps([15, 5])
        #self.final = nn.Linear(5, 1)
        self.cheby_params = torch.nn.Parameter(torch.empty([cheby_degree, 1]).normal_(mean = 0, std = 1/np.sqrt(cheby_degree*np.pi/2)), requires_grad = True)
        self.max_num_intervals = [25, 25, 25]
        self.range = abs(square[1]- square[0])
        self.min = min(square)

        self.freeze_persistence = False

    def forward(self, mb):

        L = len(mb)
        if self.freeze_persistence :
            features = torch.zeros([L, 18])
            for i in range(L):
                features[i] = mb[i]['vectors']

        else:
            births = torch.zeros([L, 3, max(self.max_num_intervals)])
            deaths = torch.zeros([L, 3, max(self.max_num_intervals)])

            for i in range(L):

                datum = mb[i]
                #f = torch.matmul(datum['chebyshev'], self.cheby_params).flatten()/torch.norm(self.cheby_params, p=2)
                f = (torch.matmul(datum['chebyshev'], self.cheby_params).flatten() - self.min)/self.range
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

            features = self.features(births, deaths)
            features =  torch.reshape(features, [L, -1])
            for i in range(L):
                mb[i]['vectors'] = features[i].detach().clone().unsqueeze(0)
            del births, deaths

        return self.project(features)

class ModelStatsRBF(nn.Module):

    def __init__(self, rbf = 10, mn = 0 , mx = 1):

        super(ModelStatsRBF, self).__init__()

        self.rbfweights = nn.Parameter(torch.empty(rbf, 1).normal_(mean = 0, std = 1/np.sqrt(rbf*np.pi/2)), requires_grad = True)
        #self.rbfweights = nn.Parameter(torch.empty(rbf, 1).uniform_(0,1), requires_grad = True)
        #self.rbfweights = nn.Sequential(nn.Linear(rbf, rbf), nn.ReLU(), nn.Linear(rbf,1))
        self.features = DiagramStats()
        self.max_num_intervals = [100, 100, 100]
        #self.project = nn.Sequential(nn.BatchNorm1d(18, affine = False),nn.Linear(18, 18), nn.Dropout(0.5), nn.ReLU(), nn.Linear(18, 1))
        self.project = nn.Sequential(nn.BatchNorm1d(18, affine = False),nn.Linear(18, 18), nn.ReLU(), nn.BatchNorm1d(18, affine = False),nn.Linear(18, 1))
        self.freeze_persistence = False
        self.mn = min(mn, mx)
        self.mx = max(mn, mx)
        self.range = self.mx - self.mn

    def forward(self, mb):
        L = len(mb)
        if self.freeze_persistence :
            features = torch.zeros([L, 18])
            for i in range(L):
                features[i] = mb[i]['vectors']
        else:

            births = torch.zeros([L, 3, max(self.max_num_intervals)])
            deaths = torch.zeros([L, 3, max(self.max_num_intervals)])
            for i in range(L):
                datum = mb[i]
                f = torch.matmul(datum['secondary_gram'], self.rbfweights).flatten()
                f  = (f - self.mn)/self.range
                #f = torch.flatten(self.rbfweights(datum['secondary_gram']))
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
            features = self.features(births, deaths)
            features =  torch.reshape(features, [L, -1])
            for i in range(L):
                mb[i]['vectors'] = features[i].detach().clone().unsqueeze(0)
            del births, deaths

        return self.project(features)

class ModelStatsRBF_double(nn.Module):

    def __init__(self, rbf = 10):

        super(ModelStatsRBF_double, self).__init__()

        #self.rbfweights = nn.Parameter(torch.empty(rbf, 1).normal_(mean = 0, std = 1/np.sqrt(rbf*np.pi/2)), requires_grad = True)
        self.rbfweights = nn.Parameter(torch.empty(rbf, 1).uniform_(0,1), requires_grad = True)
        self.features = DiagramStats()
        self.max_num_intervals = [100, 100, 100]
        self.project = nn.Sequential(nn.BatchNorm1d(36, affine = False), nn.Dropout(0.5), nn.Linear(36, 1))

    def forward(self, mb):
        L = len(mb)

        features = torch.zeros([L, 36])

        births = torch.zeros([L, 3, max(self.max_num_intervals)])
        deaths = torch.zeros([L, 3, max(self.max_num_intervals)])
        for i in range(L):

            datum = mb[i]

            features[i, 0:18] = datum['hks']
            f = torch.matmul(datum['secondary_gram'], self.rbfweights).flatten()

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
        features[:, 18:] = torch.reshape(self.features(births, deaths), [L, -1])

        return self.project(features)


class ModelRBFPL(nn.Module):

    def __init__(self, rbf = 10, resolution = 10, layers = 3, pslevel = 3,lims = [0.0, 1.0]):

        super(ModelRBFPL, self).__init__()

        #self.rbfweights = nn.Parameter(torch.empty(rbf, 1).normal_(mean = 0, std = 1/np.sqrt(rbf*np.pi/2)), requires_grad = True)
        self.rbfweights = nn.Parameter(torch.empty(rbf, 1).uniform_(-1,1), requires_grad = True)
        self.resolution = resolution
        self.layers = layers
        self.PL = PersistenceLandscapes(self.resolution, self.layers)

        self.pslevel = pslevel
        self.s = iisignature.prepare(layers, pslevel)
        self.siglength = iisignature.logsiglength(layers, pslevel)
        self.lims = lims
        print(self.siglength)
        butt = int(3* self.siglength)
        #self.fc = nn.Sequential(nn.BatchNorm1d(butt, affine= False), nn.Linear(butt, self.siglength), nn.ReLU(), nn.BatchNorm1d(self.siglength, affine = False), nn.Linear(self.siglength, 1))
        self.fc = nn.Sequential(nn.BatchNorm1d(butt, affine= False), nn.Linear(butt, butt), nn.ReLU(), nn.BatchNorm1d(butt, affine = False), nn.Linear(butt, 1))

        self.max_num_intervals = [100, 100, 100]

    def forward(self, mb):
        L = len(mb)
        births = torch.zeros([L, 3, max(self.max_num_intervals)])
        deaths = torch.zeros([L, 3, max(self.max_num_intervals)])

        for i in range(L):
            datum = mb[i]
            f = (torch.matmul(datum['secondary_gram'], self.rbfweights).flatten() - self.lims[0])/(self.lims[1] - self.lims[0])

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
        PLs = self.PL(births, deaths)
        del births, deaths

        x = torch.reshape(PLs, [L, 3, self.layers, -1])
        x = torch.transpose(x, -1, -2)
        x = LogSigFn.apply(x, self.s)
        x = self.fc(torch.reshape(x, [L, -1]))

        return x


class ModelChebyshevStats(nn.Module):

    def __init__(self, cheby_degree = 6):

        super(ModelChebyshevStats, self).__init__()

        self.cheby_params = torch.nn.Parameter(torch.empty([cheby_degree, 1]).normal_(mean = 0, std = 1/np.sqrt(cheby_degree*np.pi/2)), requires_grad = True)
        #self.cheby_params = torch.nn.Parameter(torch.tensor([ 0.12796638, -0.2425746 ,  0.20752221, -0.15955385,  0.11156322, -0.07059596,  0.04106911, -0.02148135,  0.01064266, -0.0046641 , 0.0020595 ]), requires_grad = True)
        self.max_num_intervals = [50 ,50, 50]
        self.features = DiagramStats()
        self.project = nn.Sequential(nn.BatchNorm1d(18, affine = False), nn.Linear(18,1))

    def forward(self, mb):

        L = len(mb)
        births = torch.zeros([L, 3, max(self.max_num_intervals)])
        deaths = torch.zeros([L, 3, max(self.max_num_intervals)])

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
        features = self.features(births, deaths)
        features = torch.reshape(features, [L, -1])
        del births, deaths

        return self.project(features)




class PiecewiseMonotonic(nn.Module):

    def __init__(self, planes, groups, sign = -1):

        super(PiecewiseMonotonic, self).__init__()

        #self.maxlayer = nn.MaxPool1d(groups,groups)
        self.planes = planes #planes per group
        self.groups = groups
        self.sign = sign
        #
        self.weights = nn.Parameter(torch.empty(1, planes*groups).uniform_(0, 2), requires_grad = True)
        #
        if sign <0 :
            self.bias = nn.Parameter(torch.empty(1, planes*groups).uniform_(4, 5), requires_grad = True)
        else:
            self.bias = nn.Parameter(torch.empty(1, planes*groups).uniform_(0, 1), requires_grad = True)
        #self.weights = nn.Parameter(torch.empty(1, planes*groups).uniform_(np.log(1/2), np.log(2)), requires_grad = True)
        #self.bias = nn.Parameter(torch.empty(1, planes*groups).uniform_(1, 2), requires_grad = True)
        #self.weights = torch.nn.Parameter(2*torch.rand([1, planes*groups]), requires_grad = True)
        #self.bias = torch.nn.Parameter(torch.rand([1, planes*groups])+ 1, requires_grad = True)
        self.softplus = nn.Softplus()

        self.ffmp = nn.Sequential(nn.Linear(1, planes), nn.ReLU(), nn.Linear(planes, groups), nn.ReLU(), nn.Linear(groups, 1), nn.BatchNorm1d(1, affine = False))

    def forward(self, x):
        #x = self.sign*x * torch.exp(self.weights)  + self.bias

        ##x = self.maxlayer(x.unsqueeze(1))
        ##x = torch.min(x, axis = 2).values

        #x= x.reshape([x.shape[0], self.groups, self.planes])
        #x = torch.logsumexp(x, dim = 2) #softmax over each group
        #x = -torch.logsumexp(-x, dim = 1)

        #return x
        return self.ffmp(x)

class ModelStats(nn.Module):

    def __init__(self, cheby_degree = 5, square = [0,1]):

        super(ModelStats, self).__init__()
        self.features = DiagramStats()
        #self.project = nn.Linear(18, 1)
        #self.project = feed_forward_mlps([15, 5])
        #self.final = nn.Linear(5, 1)
        self.project = nn.Sequential(nn.BatchNorm1d(18, affine = False), nn.Linear(18, 18), nn.ReLU(), nn.BatchNorm1d(18, affine = False), nn.Dropout(0.5), nn.Linear(18, 1))
        self.cheby_params = torch.nn.Parameter(torch.empty([cheby_degree, 1]).normal_(mean = 0, std = 1/np.sqrt(cheby_degree*np.pi/2)), requires_grad = True)
        #dummy = torch.zeros([cheby_degree])
        #dummy[0] = -1
        #dummy[1] = 1/2
        #self.cheby_params = torch.nn.Parameter(dummy.T, requires_grad = True)
        self.max_num_intervals = [25, 25, 25]
        self.range = abs(square[0] - square[1])
        self.min = min(square)

    def forward(self, mb):

        L = len(mb)
        births = torch.zeros([L, 3, max(self.max_num_intervals)])
        deaths = torch.zeros([L, 3, max(self.max_num_intervals)])

        for i in range(L):

            datum = mb[i]
            #f = torch.matmul(datum['chebyshev'], self.cheby_params).flatten()/torch.norm(self.cheby_params, p=2)
            f = (torch.matmul(datum['chebyshev'], self.cheby_params).flatten() - self.min)/self.range

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
        features = self.features(births, deaths)
        del births, deaths
        return self.project(torch.reshape(features, [L, -1]))

class ModelStatsMonotone(nn.Module):

    def __init__(self, planes = 30, groups = 10, sign = -1):

        super(ModelStatsMonotone, self).__init__()

        self.monotone = PiecewiseMonotonic(planes, groups, sign)
        self.features = DiagramStats()
        self.max_num_intervals = [20, 20, 20]
        #self.project = nn.Sequential(nn.BatchNorm1d(18, affine = False), nn.Linear(18, 36),nn.ReLU(), nn.BatchNorm1d(36, affine = False), nn.Linear(36, 1))
        self.project = nn.Sequential(nn.BatchNorm1d(18, affine = False), nn.Linear(18, 1))
        #self.project = nn.Sequential(nn.BatchNorm1d(18, affine = False), nn.Linear(18, 36), nn.ReLU(), nn.BatchNorm1d(36), nn.Linear(36, 1))
        #self.project = nn.Linear(18, 1)
        #self.project = nn.Sequential(nn.BatchNorm1d(12), nn.Linear(12,36),nn.LeakyReLU(0.2),nn.BatchNorm1d(36), nn.Linear(36, 1) )
        #self.t = nn.Parameter(torch.tensor(10.0), requires_grad = True)
        #self.max_num_intervals = [40, 40, 40]
        #hks data
        #self.weight = nn.Parameter(torch.tensor(0.0), requires_grad = True)
        #self.sigmoid = nn.Sigmoid()
        #self.projectfixed = nn.Sequential(nn.BatchNorm1d(18, affine = False), nn.Linear(18, 36),nn.ReLU(), nn.BatchNorm1d(36, affine = False), nn.Linear(36, 1))
        #self.projectfixed = nn.Sequential(nn.BatchNorm1d(18, affine = False), nn.Linear(18, 1))
    def forward(self, mb):

        L = len(mb)
        births = torch.zeros([L, 3, max(self.max_num_intervals)])
        deaths = torch.zeros([L, 3, max(self.max_num_intervals)])
        #fixed = torch.zeros([L, 18])
        #adaptive = torch.zeros([L, 18])
        for i in range(L):
            datum = mb[i]
            #fixed[i,:] = datum['hks_stats']
            g = self.monotone(datum['eigenvalues'])
            #g = torch.exp(-self.t * datum['eigenvalues'])
            f = torch.matmul(datum['eigenvectors_sq'], g).flatten()
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
                    #p = torch.abs(d-b)
                    #p, indices = torch.sort(p,descending = True)
                    #adaptive[i,k*6:k*6 + len( p[0:6])] = torch.cumsum(p[0:6], dim = 0)
                    #adaptive[i,k*6:k*6 + len( p[0:6])] = p[0:6]

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
        features = torch.reshape(self.features(births, deaths), [L, -1])

        #random_feats = self.elm_nonlin(torch.matmul(features, self.weight) + self.bias)
        #del births, deaths

        adaptive = self.project(features)
        #adaptive = self.project(adaptive)
        #fixed = self.projectfixed(fixed)
        #t = self.sigmoid(self.weight)

        #return t*adaptive + (1- t)*fixed
        return(adaptive)
        #return self.project(random_feats)

class MonotoneSig(nn.Module):

    def __init__(self, pslevel= 2, planes = 3, groups = 5):

        super(MonotoneSig, self).__init__()

        self.monotone = PiecewiseMonotonic(planes, groups, -1)
        self.max_num_intervals = [25, 25, 25]
        self.s = iisignature.prepare(2, pslevel)
        self.pslevel = pslevel
        self.siglength = iisignature.logsiglength(2, pslevel)
        self.final_vec_length = int(3* self.siglength)
        self.project = nn.Linear(self.final_vec_length,1)



    def forward(self, mb):
        '''
        Each data point is a tuple ('eigenvectors_sq', 'eigenvalues','simplex_tree', 'f0', 'path', 'persistence', 'filename', 'label')

        '''
        L = len(mb)
        x = torch.zeros([L, 3, max(self.max_num_intervals), 2])

        for i in range(L):
            datum = mb[i]
            g = self.monotone(datum['eigenvalues'])
            f = torch.matmul(datum['eigenvectors_sq'], g).flatten()

            birthv = [[],[],[]]
            deathv = [[],[],[]]
            #recompute persistence
            datum['simplex_tree'] = filtration_update(datum['simplex_tree'], f)
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
                b = f[birthv[k]].flatten()
                d = f[deathv[k]].flatten()
                p = torch.abs(d - b)
                l = min(len(b), self.max_num_intervals[k])
                pathorder = list(torch.sort(p,descending=True).indices[0:l].detach().numpy())
                x[i, k, 0:l, 0] = b[pathorder[::-1]]
                x[i, k, 0:l, 1] = d[pathorder[::-1]]
                #x[i, k, 0:l, 2] = p[pathorder]


                if l < self.max_num_intervals[k] and l > 0:
                    x[i, k, l: , 0] = b[pathorder[0]].detach()
                    x[i, k, l: , 1] = d[pathorder[0]].detach()
                    #x[i, k, 2, l : ] = p[-1].detach()

                #del b, d, p
            del birthv, deathv

        x = LogSigFn.apply(x, self.s)
        x = x.reshape([L, -1])
        return self.project(x)

class ModelMonotonic(nn.Module):

    def __init__(self, resolution = 20, sigma = 0.1, planes = 30, groups = 10, square = [0.0, 1.0]):

        super(ModelMonotonic, self).__init__()

        self.monotone = PiecewiseMonotonic(planes, groups, -1)
        self.max_num_intervals = [25 ,25, 25]
        self.square = square
        self.sigma = sigma
        self.resolution = resolution
        self.PI = PersistenceImage(self.resolution, self.sigma, self.square)
        self.averager = nn.Sequential(nn.Conv2d(3, 20, 2, 1), nn.BatchNorm2d(20)) #2*20*20 -> 2*19*19
        dimension = int((self.resolution -  1))
        self.final_fc = feed_forward_mlps((20*dimension**2, 20))
        self.final_final_fc = nn.Sequential(nn.Linear(20, 1), nn.BatchNorm1d(1))

    def forward(self, mb):

        L = len(mb)
        births = torch.zeros([L, 3, max(self.max_num_intervals)])
        deaths = torch.zeros([L, 3, max(self.max_num_intervals)])

        for i in range(L):

            datum = mb[i]
            g = self.monotone(datum['eigenvalues'])
            f = torch.matmul(datum['eigenvectors_sq'], g).flatten()
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
        del births, deaths

        x = self.averager(PIs)
        x = self.final_fc(x.flatten(1))
        x = self.final_final_fc(x)

        return x

class ModelControl(nn.Module):

    def __init__(self, resolution = 20, sigma = 0.1):

        super(ModelControl, self).__init__()

        self.max_num_intervals = [25, 25, 25, 25]
        self.sigma = sigma
        self.resolution = resolution
        #self.PI = nn.ModuleList([PersistenceImage(self.resolution, self.sigma)]*4)
        #self.averager = nn.Sequential(nn.Conv2d(4, 40, 2, 1, groups=4), nn.BatchNorm2d(40)) #2*20*20 -> 2*19*19
        #dimension = int((self.resolution -  1))
        #self.do = nn.Dropout2d(p=0.5)
        #self.final_fc = feed_forward_mlps((40*dimension**2, 50))
        #self.final_final_fc = nn.Sequential(nn.Linear(50, 1), nn.BatchNorm1d(1))

        self.PI = PersistenceImage(self.resolution, self.sigma)
        self.averager = nn.Sequential(nn.Conv2d(3, 20, 2, 1), nn.BatchNorm2d(20)) #2*20*20 -> 2*19*19
        dimension = int((self.resolution -  1))
        self.final_fc = feed_forward_mlps((20*dimension**2, 20))
        self.final_final_fc = nn.Sequential(nn.Linear(20, 1), nn.BatchNorm1d(1))

    def forward(self, mb):
        '''
        Each data point is a L*4*2*50 tensor, where L = len(mini batch), 4 = number of diagrams
        '''
        shape = mb.shape
        mb = torch.reshape(mb, [shape[0]*shape[1], 2, shape[3]])
        PIs = self.PI(mb[:,0,:], mb[:,1,:])

        x = self.averager(torch.reshape(PIs, [shape[0], shape[1], self.resolution, self.resolution]))
        #x = self.do(x)
        x = self.final_fc(x.flatten(1))
        x = self.final_final_fc(x)

        return x

class ModelControlStats(nn.Module):

    def __init__(self):

        super(ModelControlStats, self).__init__()
        self.features = DiagramStats()
        self.project = nn.Sequential(nn.BatchNorm1d(15), nn.Linear(15, 30),nn.ReLU(), nn.BatchNorm1d(30), nn.Linear(30, 1) )
        #self.project = nn.Sequential(nn.BatchNorm1d(15), nn.Linear(15, 1))

    def forward(self, mb):
        '''
        Each data point is a L*4*2*50 tensor, where L = len(mini batch), 4 = number of diagrams
        '''
        shape = mb.shape
        mb = torch.reshape(mb, [shape[0]*shape[1], 2, shape[3]])
        features = self.features(mb[:,0,:], mb[:,1,:])
        return self.project(torch.reshape(features, [shape[0], -1]))

class LinearModel(nn.Module):

    def __init__(self, length = 18):

        super(LinearModel, self).__init__()
        #self.project = nn.Sequential(nn.BatchNorm1d(18, affine = False), nn.Linear(18, 1))
        #self.project = nn.Sequential(nn.BatchNorm1d(18), nn.Dropout(0.25), nn.Linear(18, 36),nn.ReLU(),nn.Dropout(0.25), nn.Linear(9, 1) )
        #self.project = nn.Sequential(nn.BatchNorm1d(18, affine = False), nn.Linear(18, 18), nn.ReLU(), nn.BatchNorm1d(18, affine = False), nn.Linear(18,1))
        #self.project = nn.Sequential(nn.BatchNorm1d(length, affine = False),nn.Linear(length, length ), nn.ReLU(), nn.BatchNorm1d(length ), nn.Linear(length , 1))
        self.project = nn.Sequential(nn.BatchNorm1d(length, affine = False), nn.Linear(length, length), nn.ReLU(),  nn.BatchNorm1d(length ), nn.Dropout(0.5), nn.Linear(length , 1))

    def forward(self, mb):

        return self.project(mb)

class ModelSupSample(nn.Module):

    def __init__(self, resolution = 20, sigma = 0.1, error_tol = 0):

        super(ModelSupSample, self).__init__()

        self.layer_units = (1, 20, 20, 20, 1)
        self.ev_function = feed_forward_mlps(self.layer_units, bn = False)
        #self.rescaler = nn.Sigmoid()

        self.max_num_intervals = [30, 30, 30, 30]
        self.bounder = nn.ReLU()

        self.sigma = sigma
        self.resolution = resolution
        self.oversample_res = 2*self.resolution + 2
        self.PI = PersistenceImage(self.oversample_res, self.sigma)
        self.averager = nn.Sequential(nn.AvgPool2d(4, stride=2), nn.Conv2d(4, 20, 2, 1), nn.BatchNorm2d(20)) #2*20*20 -> 2*19*19
        dimension = int((self.resolution -  1))
        self.final_fc = feed_forward_mlps((20*dimension**2, 20))
        self.final_final_fc = nn.Sequential(nn.Linear(20, 1), nn.BatchNorm1d(1))
        self.update_counter = 0
        self.error_tol = error_tol

    def forward(self, mb):
        '''
        Each data point is a tuple ('eigenvectors_sq', 'eigenvalues', 'eigenvalue_quantiles' 'simplex_tree', 'f0','hks', 'birth', 'persistence', 'filename', 'label')

        '''
        L = len(mb)
        births = torch.zeros([L, 4, max(self.max_num_intervals)])
        persistence = torch.zeros([L, 4, max(self.max_num_intervals)])
        regulariser = torch.zeros([1])
        for i in range(L):
            datum = mb[i]
            #f = self.rescaler(torch.matmul(datum['eigenvectors_sq'],self.ev_function(datum['eigenvalues']))).flatten()
            f = torch.matmul(datum['eigenvectors_sq'],self.ev_function(datum['eigenvalues'])).flatten()
            #regulariser += torch.sum(self.bounder(-f) + self.bounder(f-1))/len(f)


            if torch.max(torch.abs(f - datum['f0'])) <= self.error_tol and self.training: #np.random.uniform(low =0.0, high = 2.0*self.error_tol)
                for k in range(4):
                    b, p = datum['birth'][k], datum['persistence'][k]
                    births[i, k,0:len(b)] = b
                    persistence[i, k, 0:len(p)] = p

            else: #deviation too much, update persistence
                if self.training:
                    self.update_counter += 1
                    datum['f0'] = f.detach()
                datum['birth'] = []
                datum['persistence'] = []
                birthv = [[],[],[],[]]
                deathv = [[],[],[],[]]
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
                            if 1000 in interval[1]:
                                k = 0 #sort into diagram label
                                #regulariser += self.bounder(-f[bv] - 2*self.sigma)**2 + self.bounder(f[bv] - 2*self.sigma)**2 + self.bounder(f[dv] - 1- 2*self.sigma)**2 + self.bounder(-f[dv] + 1 - 2*self.sigma )**2
                                regulariser += (f[bv]-0.5*self.sigma)**2 + (f[dv]-1+0.5*self.sigma)**2
                            else:
                                k = 1
                        if len(interval[0]) == 2: #H1
                            dv = max([v for v in interval[0] if v != 1000], key = lambda v: f[v])
                            bv = min([v for v in interval[1] if v != 1000], key = lambda v: f[v])
                            if 1000 in interval[0]:
                                k = 2
                            else:
                                k = 3
                    birthv[k].append(bv)
                    deathv[k].append(dv)

                for k in range(4):
                    b = f[birthv[k]]
                    p = torch.abs(f[birthv[k]] - f[deathv[k]])
                    if len(b) > self.max_num_intervals[k]:
                        truncation = torch.topk(p, self.max_num_intervals[k], largest=True, sorted=False).indices
                        b = b[truncation]
                        p = p[truncation]
                    births[i, k,0:len(b)] = b
                    persistence[i, k,0:len(p)] = p

                    datum['birth'].append(b.detach())
                    datum['persistence'].append(p.detach())
                    del b, p
                mb[i] = datum
                del birthv, deathv
                #pickle.dump(datum, open(datum['filename'], 'wb'))

        births = torch.reshape(births, [L*4, -1])
        persistence = torch.reshape(persistence, [L*4, -1])
        PIs = self.PI(births, persistence)
        PIs = torch.reshape(PIs, [L, 4, self.oversample_res, self.oversample_res])
        del births, persistence

        x = self.averager(PIs)
        #x = self.do(x)
        x = self.final_fc(x.flatten(1))
        x = self.final_final_fc(x)

        return x, regulariser/L

class ModelChebyshev(nn.Module):

    def __init__(self, resolution = 20, sigma = 0.1, error_tol = 0, cheby_degree = 11, square = [0.0, 1.0]):

        super(ModelChebyshev, self).__init__()
        self.freeze_persistence = False
        self.cheby_params = torch.nn.Parameter(torch.empty([cheby_degree, 1]).normal_(mean = 0, std = 1/np.sqrt(cheby_degree*np.pi/2)), requires_grad = True)
        #self.cheby_params = torch.nn.Parameter(torch.tensor([ 0.12796638, -0.2425746 ,  0.20752221, -0.15955385,  0.11156322, -0.07059596,  0.04106911, -0.02148135,  0.01064266, -0.0046641 , 0.0020595 ]), requires_grad = True)
        self.max_num_intervals = [25 ,25, 25]
        self.square = square
        self.sigma = 1/(resolution-3)
        self.resolution = resolution
        self.PI = PersistenceImage(self.resolution, self.sigma)
        self.averager = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(in_channels = 3,out_channels = 3, kernel_size = 4, stride = 2), nn.ReLU(), nn.BatchNorm2d(3), nn.Conv2d(in_channels = 3,out_channels = 1, kernel_size = 4, stride = 2), nn.ReLU(), nn.BatchNorm2d(1))
        butt = ((self.resolution // 2 -1)//2 -1)**2
        self.final_fc = nn.Linear(butt, 1)
        #self.averager = nn.Sequential(nn.Conv2d(3, 20, 2, 1), nn.BatchNorm2d(20)) #2*20*20 -> 2*19*19
        #dimension = int((self.resolution -  1))
        #self.averager = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(3,1, 1, 1))
        #dimension = self.resolution
        #self.averager = nn.Sequential(nn.BatchNorm2d(3), nn.Conv2d(in_channels = 3,out_channels = 3, kernel_size = 4, stride = 2), nn.ReLU(), nn.BatchNorm2d(3), nn.Conv2d(in_channels = 3,out_channels = 1, kernel_size = 4, stride = 2), nn.ReLU(), nn.BatchNorm2d(1))
        #self.averager = nn.Sequential(nn.Conv2d(in_channels = 3,out_channels = 3, kernel_size = 2, stride = 1, bias = False), nn.ReLU(), nn.MaxPool2d(kernel_size = 4, stride = 2, padding = 2), nn.Conv2d(in_channels = 3,out_channels = 1, kernel_size = 2, stride = 1, bias = False), nn.ReLU())

        #butt = ((self.resolution-1)//2)**2
        #self.final_fc = feed_forward_mlps((20*dimension**2, 20))
        #self.final_final_fc = nn.Sequential(nn.Linear(20, 1), nn.BatchNorm1d(1))
        #butt = dimension**2
        #self.final_fc = nn.Sequential(nn.ReLU(), nn.BatchNorm1d(butt), nn.Linear(butt, 1))

        self.lims = square

    def forward(self, mb):

        L = len(mb)

        if not self.freeze_persistence:

            births = torch.zeros([L, 3, max(self.max_num_intervals)])
            deaths = torch.zeros([L, 3, max(self.max_num_intervals)])

            for i in range(L):

                datum = mb[i]
                f = (torch.matmul(datum['chebyshev'], self.cheby_params).flatten() - self.lims[0])/(self.lims[1] - self.lims[0])
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
            for i in range(L):
                mb[i]['images'] = PIs[i].detach().clone().unsqueeze(0)
            del births, deaths
        else:
            PIs = torch.zeros([L, 3, self.resolution, self.resolution])
            for i in range(L):
                PIs[i] = mb[i]['images']

        x = self.averager(PIs)
        x = self.final_fc(x.flatten(1))
        #x = self.final_final_fc(x)

        return x

class ModelChebyshevLS(nn.Module):

    def __init__(self, resolution = 20, layers = 3, cheby_degree = 11, square = [0, 1], pslevel = 3):

        super(ModelChebyshevLS, self).__init__()

        self.cheby_params = torch.nn.Parameter(torch.empty([cheby_degree, 1]).normal_(mean = 0, std = 1/np.sqrt(cheby_degree*np.pi/2)), requires_grad = True)
        self.max_num_intervals = [25 ,25, 25]
        self.resolution = resolution
        self.lims = square
        self.layers = layers
        self.PL = PersistenceLandscapes(self.resolution, self.layers)
        #butt = 3*self.resolution *self.layers
        #self.fc = nn.Sequential(nn.BatchNorm1d(butt), nn.Linear(butt, butt), nn.ReLU(), nn.BatchNorm1d(butt), nn.Linear(butt, 1))
        #self.fc = nn.Sequential(nn.BatchNorm1d(butt, affine = False), nn.Linear(butt, 1))

        self.pslevel = pslevel
        self.s = iisignature.prepare(layers-1, pslevel)
        self.siglength = iisignature.logsiglength(layers-1, pslevel)
        print(self.siglength)
        butt = int(3* self.siglength)
        self.fc = nn.Sequential(nn.BatchNorm1d(butt, affine= False), nn.Linear(butt, butt), nn.ReLU(), nn.BatchNorm1d(butt, affine = False), nn.Linear(butt, 1))
        #self.fc = nn.Sequential(nn.BatchNorm1d(butt, affine= False), nn.Linear(butt, self.siglength), nn.ReLU(), nn.BatchNorm1d(self.siglength, affine = False), nn.Linear(self.siglength, 1))
        #self.fc = nn.Sequential(nn.BatchNorm1d(butt),nn.Linear(butt, 1))

    def forward(self, mb):

        L = len(mb)
        births = torch.zeros([L, 3, max(self.max_num_intervals)])
        deaths = torch.zeros([L, 3, max(self.max_num_intervals)])

        for i in range(L):

            datum = mb[i]
            f = (torch.matmul(datum['chebyshev'], self.cheby_params).flatten() - self.lims[0])/(self.lims[1] - self.lims[0])
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
        PLs = self.PL(births, deaths)[:, 1:3, :]
        del births, deaths
        #x = self.fc(torch.reshape(PLs, [L, -1]))
        x = torch.reshape(PLs, [L, 3, self.layers-1, -1])
        x = torch.transpose(x, -1, -2)
        x = LogSigFn.apply(x, self.s)
        x = self.fc(torch.reshape(x, [L, -1]))
        return x

class ModelChebyshevSW(nn.Module):

    def __init__(self, resolution = 5, cheby_degree = 11, square = [0, 1], pslevel = 3):

        super(ModelChebyshevSW, self).__init__()

        self.cheby_params = torch.nn.Parameter(torch.empty([cheby_degree, 1]).normal_(mean = 0, std = 1/np.sqrt(cheby_degree*np.pi/2)), requires_grad = True)
        #self.chebnet = nn.Sequential(nn.Linear(cheby_degree, cheby_degree), nn.ReLU(), nn.Linear(cheby_degree, 1))
        self.max_num_intervals = [25 ,25, 25]
        self.resolution = resolution
        self.lims = square
        self.SW = SlicedWasserstein(self.resolution)
        self.freeze_persistence = False

        self.pslevel = pslevel
        self.s = iisignature.prepare(resolution, pslevel)
        self.siglength = iisignature.logsiglength(resolution, pslevel)
        #print(self.siglength)
        butt = int(3* self.siglength)
        #self.fc = nn.Sequential(nn.BatchNorm1d(butt, affine= False), nn.Linear(butt, butt), nn.ReLU(), nn.BatchNorm1d(butt, affine = False), nn.Dropout(0.5), nn.Linear(butt, 1))
        self.fc = nn.Sequential(nn.BatchNorm1d(butt, affine= False), nn.Linear(butt, 1))

    def forward(self, mb):

        L = len(mb)
        sigs = torch.zeros([L, 3, self.siglength])
        norm = max(self.cheby_params.norm(2), 1e-4)
        c = self.cheby_params / norm
        if self.freeze_persistence:
            for i in range(L):
                sigs[i] = mb[i]['sigs']
        else:

            for i in range(L):

                datum = mb[i]
                #f = (torch.matmul(datum['chebyshev'], c).flatten() - self.lims[0])/(self.lims[1] - self.lims[0])
                f = torch.matmul(datum['chebyshev'], c).flatten()

                #f = torch.flatten(self.chebnet(datum['chebyshev']))
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
                        sw = self.SW(b,d)
                        v = LogSigFn.apply(sw, self.s)
                        sigs[i,k] = v
                        del b, d

                mb[i]['sigs'] = sigs[i].detach().clone()

                del birthv, deathv

        x = self.fc(torch.reshape(sigs, [L, -1]))


        return x

class ModelRank(nn.Module):

    def __init__(self, resolution = 20, sigma = 0.1, error_tol = 0):

        super(ModelRank, self).__init__()

        self.rank_params = torch.nn.Parameter(2*torch.rand([20, 1]), requires_grad = True)
        #self.rescaler = nn.Sigmoid()

        self.max_num_intervals = [30, 30, 30, 30]
        self.bounder = nn.ReLU()

        self.sigma = sigma
        self.resolution = resolution
        self.oversample_res = 2*self.resolution + 2
        self.PI = PersistenceImage(self.oversample_res, self.sigma)
        self.averager = nn.Sequential(nn.AvgPool2d(4, stride=2), nn.Conv2d(4, 20, 2, 1), nn.BatchNorm2d(20)) #2*20*20 -> 2*19*19
        dimension = int((self.resolution -  1))
        self.final_fc = feed_forward_mlps((20*dimension**2, 20))
        self.final_final_fc = nn.Sequential(nn.Linear(20, 1), nn.BatchNorm1d(1))
        self.update_counter = 0
        self.error_tol = error_tol

    def forward(self, mb):
        '''
        Each data point is a tuple ('eigenvectors_sq', 'eigenvalues', 'eigenvalue_rank', 'simplex_tree', 'f0','hks', 'birth', 'persistence', 'filename', 'label')

        '''
        L = len(mb)
        births = torch.zeros([L, 4, max(self.max_num_intervals)])
        persistence = torch.zeros([L, 4, max(self.max_num_intervals)])
        regulariser = torch.zeros([1])

        for i in range(L):

            datum = mb[i]
            f = torch.matmul(datum['eigenvectors_sq'], self.rank_params).flatten()

            if torch.max(torch.abs(f - datum['f0'])) <= self.error_tol and self.training: #np.random.uniform(low =0.0, high = 2.0*self.error_tol)
                for k in range(4):
                    b, p = datum['birth'][k], datum['persistence'][k]
                    births[i, k,0:len(b)] = b
                    persistence[i, k, 0:len(p)] = p

            else: #deviation too much, update persistence
                if self.training:
                    self.update_counter += 1
                    datum['f0'] = f.detach()
                datum['birth'] = []
                datum['persistence'] = []
                birthv = [[],[],[],[]]
                deathv = [[],[],[],[]]
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
                            if 1000 in interval[1]:
                                k = 0 #sort into diagram label
                                regulariser += (f[bv])**2 + (f[dv]-1)**2
                            else:
                                k = 1
                        if len(interval[0]) == 2: #H1
                            dv = max([v for v in interval[0] if v != 1000], key = lambda v: f[v])
                            bv = min([v for v in interval[1] if v != 1000], key = lambda v: f[v])
                            if 1000 in interval[0]:
                                k = 2
                            else:
                                k = 3
                    birthv[k].append(bv)
                    deathv[k].append(dv)

                for k in range(4):
                    b = f[birthv[k]]
                    p = torch.abs(f[birthv[k]] - f[deathv[k]])
                    if len(b) > self.max_num_intervals[k]:
                        truncation = torch.topk(p, self.max_num_intervals[k], largest=True, sorted=False).indices
                        b = b[truncation]
                        p = p[truncation]
                    births[i, k,0:len(b)] = b
                    persistence[i, k,0:len(p)] = p

                    datum['birth'].append(b.detach())
                    datum['persistence'].append(p.detach())
                    del b, p
                mb[i] = datum
                del birthv, deathv
                #pickle.dump(datum, open(datum['filename'], 'wb'))

        births = torch.reshape(births, [L*4, -1])
        persistence = torch.reshape(persistence, [L*4, -1])
        PIs = self.PI(births, persistence)
        PIs = torch.reshape(PIs, [L, 4, self.oversample_res, self.oversample_res])
        del births, persistence

        x = self.averager(PIs)
        #x = self.do(x)
        x = self.final_fc(x.flatten(1))
        x = self.final_final_fc(x)

        return x, regulariser/L

class Model(nn.Module):

    def __init__(self, resolution = 20, sigma = 0.1, error_tol = 0):

        super(Model, self).__init__()

        self.layer_units = (1, 20, 20, 20, 1)
        self.ev_function = feed_forward_mlps(self.layer_units, bn = False)
        #self.rescaler = nn.Sigmoid()

        self.max_num_intervals = [30, 30, 30, 30]
        self.bounder = nn.ReLU()

        self.sigma = sigma
        self.resolution = resolution
        self.PI = PersistenceImage(self.resolution, self.sigma)
        self.averager = nn.Sequential(nn.Conv2d(4, 20, 2, 1), nn.BatchNorm2d(20)) #2*20*20 -> 2*19*19
        dimension = int((self.resolution -  1))
        self.final_fc = feed_forward_mlps((20*dimension**2, 20))
        self.final_final_fc = nn.Sequential(nn.Linear(20, 1), nn.BatchNorm1d(1))
        self.update_counter = 0
        self.error_tol = error_tol

    def forward(self, mb):
        '''
        Each data point is a tuple ('eigenvectors_sq', 'eigenvalues', 'eigenvalue_quantiles' 'simplex_tree', 'f0','hks', 'birth', 'persistence', 'filename', 'label')

        '''
        L = len(mb)
        births = torch.zeros([L, 4, max(self.max_num_intervals)])
        persistence = torch.zeros([L, 4, max(self.max_num_intervals)])
        regulariser = torch.zeros([1])
        for i in range(L):
            datum = mb[i]
            #f = self.rescaler(torch.matmul(datum['eigenvectors_sq'],self.ev_function(datum['eigenvalues']))).flatten()
            f = torch.matmul(datum['eigenvectors_sq'],self.ev_function(datum['eigenvalues'])).flatten()
            #regulariser += torch.sum(self.bounder(-f) + self.bounder(f-1))/len(f)


            if torch.max(torch.abs(f - datum['f0'])) <= self.error_tol and self.training: #np.random.uniform(low =0.0, high = 2.0*self.error_tol)
                for k in range(4):
                    b, p = datum['birth'][k], datum['persistence'][k]
                    births[i, k,0:len(b)] = b
                    persistence[i, k, 0:len(p)] = p

            else: #deviation too much, update persistence
                if self.training:
                    self.update_counter += 1
                    datum['f0'] = f.detach()
                datum['birth'] = []
                datum['persistence'] = []
                birthv = [[],[],[],[]]
                deathv = [[],[],[],[]]
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
                            if 1000 in interval[1]:
                                k = 0 #sort into diagram label
                                #regulariser += self.bounder(-f[bv] - 2*self.sigma)**2 + self.bounder(f[bv] - 2*self.sigma)**2 + self.bounder(f[dv] - 1- 2*self.sigma)**2 + self.bounder(-f[dv] + 1 - 2*self.sigma )**2
                                regulariser += (f[bv]-0.5*self.sigma)**2 + (f[dv]-1+0.5*self.sigma)**2
                            else:
                                k = 1
                        if len(interval[0]) == 2: #H1
                            dv = max([v for v in interval[0] if v != 1000], key = lambda v: f[v])
                            bv = min([v for v in interval[1] if v != 1000], key = lambda v: f[v])
                            if 1000 in interval[0]:
                                k = 2
                            else:
                                k = 3
                    birthv[k].append(bv)
                    deathv[k].append(dv)

                for k in range(4):
                    b = f[birthv[k]]
                    p = torch.abs(f[birthv[k]] - f[deathv[k]])
                    if len(b) > self.max_num_intervals[k]:
                        truncation = torch.topk(p, self.max_num_intervals[k], largest=True, sorted=False).indices
                        b = b[truncation]
                        p = p[truncation]
                    births[i, k,0:len(b)] = b
                    persistence[i, k,0:len(p)] = p

                    datum['birth'].append(b.detach())
                    datum['persistence'].append(p.detach())
                    del b, p
                mb[i] = datum
                del birthv, deathv
                #pickle.dump(datum, open(datum['filename'], 'wb'))

        births = torch.reshape(births, [L*4, -1])
        persistence = torch.reshape(persistence, [L*4, -1])
        PIs = self.PI(births, persistence)
        PIs = torch.reshape(PIs, [L, 4, self.resolution, self.resolution])
        del births, persistence

        x = self.averager(PIs)
        #x = self.do(x)
        x = self.final_fc(x.flatten(1))
        x = self.final_final_fc(x)

        return x, regulariser/L

class ModelFixedDgms(nn.Module):

    def __init__(self, resolution = 20):

        super(ModelFixedDgms, self).__init__()

        self.resolution = resolution
        self.averager = nn.Sequential(nn.Conv2d(3, 20, 2, 1), nn.BatchNorm2d(20)) #2*20*20 -> 2*19*19
        dimension = int((self.resolution -  1))
        self.final_fc = feed_forward_mlps((20*dimension**2, 20))
        self.final_final_fc = nn.Sequential(nn.Linear(20, 1), nn.BatchNorm1d(1))

    def forward(self, mb):
        '''
        Each data point is a L*4*2*50 tensor, where L = len(mini batch), 4 = number of diagrams
        '''
        x = self.averager(mb)
        x = self.final_fc(x.flatten(1))
        x = self.final_final_fc(x)

        return x
