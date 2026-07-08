import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy.random as rng
from geomloss import SamplesLoss
import random
import time

sinkhorn = SamplesLoss(
    loss="sinkhorn",
    p=2,
    blur= 0.15
)

repul_param = 0
d = 2 
nech = 20
Nmax = 20

def repulsion_penalty3(P, tol=1e-2):
    dist = torch.cdist(P, P)  # (n, n)
    mask = ~torch.eye(dist.size(0), dtype=bool, device=P.device)
    return (torch.relu(1*(tol - dist[mask]))).mean()



#Un tirage de configuration LHD/LHS
def lhs(n_points, n_dim):
    perm = []
    for j in range(n_dim):
        perm.append(rng.permutation(n_points))
        
    return perm

#calcule d'une matrice de distance
def dist(X,Y, lx = None, ly = None):
    if(lx == None):
        lx = len(X)
    if(ly == None):  
        ly = len(Y)
    C = np.zeros([lx,ly])
    C = np.sum((X[:, None, :] - Y[None, :, :])**2, axis=2)
    return C

def borne(Y, a, b, d = 2, eps=1e-6):
    if len(a) != d:
        if len(a) == 1:
            a = torch.ones(d)*a
        else:
            a = torch.zeros(d)
            
    if len(b) != d:
        if len(b) == 1:
            b = torch.ones(d)*b
        else:
            b = torch.zeros(d)
    return -torch.mean(torch.log(Y - a + eps) + torch.log(b - Y + eps))


def bornew(Y, eps=1e-6):
    return -torch.mean( torch.log(Y + eps) +   torch.log(1.0 - Y + eps))


def min_dist(X, Y, mY):
    vals = []
    for Xid in X:
        mXid = torch.ones(1, device=X.device)  # masse uniforme
        vals.append(sinkhorn(mXid, Xid.unsqueeze(0), mY, Y))
    return torch.min(torch.stack(vals))

def gaussian_kernel(X, Y, sigma = .5):    
    D2 = torch.cdist(X, Y, p=2) **2
    return torch.exp(-D2 / (2 * sigma ** 2))



def MMD_n(X, Y,  kern = gaussian_kernel, sigma = .5, Kyy = None, Kxx = None):
    # Matrices de noyau
    if Kxx is None:
        Kxx = kern(X, X, sigma)/len(X)**2
    if Kyy is None:
        Kyy = kern(Y, Y, sigma)/len(Y)**2
    Kxy = kern(X, Y, sigma)/(len(X)*len(Y))
    return (Kxx.sum() + Kyy.sum() - 2 * Kxy.sum())



def MMD_p(X, Wx, Y, Wy, kern = gaussian_kernel, sigma = .5, Kyy = None, Kxx = None):
    # Matrices de noyau
    if Kxx is None:
        Kxx = Wx[:, None] * Wx[None,:] *kern(X, X, sigma)
    if Kyy is None:
        Kyy = Wy[:, None] * Wy[None,:]*  kern(Y, Y, sigma)
    Kxy = Wx[:, None] * Wy[None,:] * kern(X, Y, sigma)
    return (Kxx.sum() + Kyy.sum() - 2 * Kxy.sum())


def min_MMD_p(ech, W_ech, Y, W, kern, sigma = .5):
    Kyy = W[:, None] * W[None,:].T * kern(Y, Y, sigma)
    return torch.min(torch.stack([
        MMD_p(e, we, Y, W, kern, sigma, Kyy) for e, we in zip(ech, W_ech)
    ]))


def gumbel_sigmoid(logits):
     return torch.sigmoid(logits)

def get_masked_points(P, w):
     m = gumbel_sigmoid(w)
     return P, m
 
def quasimin(x, tau=0.1):
    return -tau * torch.logsumexp(-x / tau, dim=0)



def distsm(nb_point):
    return torch.sum(torch.stack([(x - y)**2 for x in nb_point for y in nb_point]))/2


def soft_count_near(x, c=0.0, temperature=10.0):
    return torch.exp(-temperature * (x - c)**2).sum()

def w_penal(w):
    return torch.exp(-w**2).sum()


def uniform_test_stat(inertias, a = 0., b = nech**(-1/d)):
    with torch.no_grad():
        a = torch.min(torch.min(inertias), torch.tensor(a))
        b = torch.max(torch.max(inertias), torch.tensor(b))
    u = (inertias - a) / (b - a)
    u_sorted, _ = torch.sort(u)

    n = u.shape[0]
    i = torch.arange(1, n + 1, device=u.device)

    # ECDF vs uniforme
    D_plus  = torch.max(i / n - u_sorted)
    D_minus = torch.max(u_sorted - (i - 1) / n)

    return torch.max(D_plus, D_minus)



def cvm_uniform_loss(inertias, a=repul_param*2, b= 1.2 * Nmax**(-1/d)):
    if torch.tensor(b)<torch.max(inertias):
        b= torch.max(inertias) + (b-a)*.01
    """if torch.tensor(a)> torch.min(inertias):
        a = torch.min(inertias) - (b-a)*.0001"""
    u = (inertias - a) / (b - a)
    u_sorted, _ = torch.sort(u)

    n = u.shape[0]
    i = torch.arange(1, n + 1, device=u.device, dtype=u.dtype)

    target = (2*i - 1) / (2*n)

    return torch.mean((u_sorted - target)**2), a, b


def test_dist_a_unif(inertias, a=-0., b= 1.2 * Nmax**(-1/d)):
    if torch.tensor(b)<torch.max(inertias):
        b= torch.max(inertias) + (b-a)*.01
    if torch.tensor(a)> torch.min(inertias):
        a = torch.min(inertias) - (b-a)*.0001
    ref = (torch.rand((len(inertias))) + a) * (b - a)
    val = sinkhorn(inertias.view(-1,1),ref.view(-1,1))
    return val, a, b


def dist2cp(X,Y, lx = None, ly = None):
    if(lx == None):
        lx = len(X)
    if(ly == None):  
        ly = len(Y)
    C = np.zeros([lx,ly])
    C = np.sum((X[:, None, :] - Y[None, :, :])**2, axis=2)
    return C

def distcp(X,Y, lx = None, ly = None):
    C = (X[:, None] - Y[None, :])**2
    return C

def kernel(x, y, sigma = .01):
    return torch.exp(- (x[:, None] - y[None, :])**2 / (2*sigma**2))



