import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy.random as rng
import outils
from geomloss import SamplesLoss
import random
import time
import affichage
import os



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé :", device)
torch.set_default_device(device)


sinkhorn = SamplesLoss(
    loss="sinkhorn",
    p=2,
    blur= 0.15
)


def LHS(nrechlhs, nech, d):
    maximindist = 0 #initialisation de notre maximum de disteance minimum
    bestperme = []  #initialisation de notre liste qui contient notre meilleur paterne
    for i in range(nrechlhs):
        
        permtemp = outils.lhs(nech, 2*d)   #tirrage d'une configuration LHS
        C = outils.dist(np.array(permtemp).T, np.array(permtemp).T)    #Calcule de la matrice de distance
        np.fill_diagonal(C, 2**20) #On cherche la distance minimum donc afin de ne pas prendre les 0 de la diagonal on met une grande valeur à celle ci
        mintemp = np.min(C) # recherche de la distance minimal
        if mintemp > maximindist: #on enregistre notre trouvaille si il est mieux que tout ce que l'on a trouver jusqu'a present
            bestperme = permtemp
            maximindist = mintemp
    return bestperme


def calc_param(nech, d, nrechlhs, Nmin, Nmax):
    esp = np.zeros((nech,d))
    var = np.zeros((nech,d))
    permu = LHS(nrechlhs, nech, d)
    rangesp = np.linspace(0, 1, nech + 1)
    for i in range(d):
        esp[:,i] = rng.uniform(rangesp[:-1], rangesp[1:])[permu[0]]
        rangevar = np.array(np.linspace(np.zeros(nech), esp[:,i] * (1 - esp[:,i]),nech + 1)).T
        for j in range(nech):
            var[j,i] = rng.uniform(rangevar[j][permu[-d + i][j]], rangevar[j][(permu[-d+i][j] + 1)])
    #Calcule de notre alpha et beta grace à notre esp et var tirer precedement


    N = np.random.randint(Nmin, (Nmax+1), nech)
    alpha = (np.array(esp) * (np.array(esp) *(1 - np.array(esp))-var))/(var)
    beta = ((1 - np.array(esp)) * (np.array(esp) *(1 - np.array(esp))-var))/(var)
    return alpha, beta, N





def constru_init(argtir, a, b, init_coin, Nmax, d):
    ech = []
    
    if init_coin:
        A = torch.rand((Nmax,d))/4
        A += a
        ech.append(A)
        A = torch.rand((Nmax,d))/4
        A[:,0] += a[0]
        A[:,1] = b[1] - A[:,1] 
        ech.append(A)
        A = torch.rand((Nmax,d))/4
        A[:,0] = b[0] - A[:,0]
        A[:,1] += a[1]
        ech.append(A)
        A = torch.rand((Nmax,d))/4
        A = b - A
        ech.append(A)

    #tirage de nos echantillon
    for alpha, beta, n_samples in argtir:
       # alpha, beta : (d,)
        alpha_t = torch.tensor(alpha, dtype=torch.float32)
        beta_t  = torch.tensor(beta, dtype=torch.float32)
        # Distribution Beta vectorisée
        dist = torch.distributions.Beta(alpha_t, beta_t)
        # Échantillonnage : (n_samples, d)
        samples = dist.sample((n_samples,))
        # Si tu veux le support [a, b]
        samples = a + (b - a) * samples
        ech.append(samples)
    return ech

def crea_ech_ini(Nmin, Nmax, nech, a = torch.tensor([0,0]), b = torch.tensor([1,1]), d= 2,nrechlhs = 60000,
                 all_opt = True, aff = True, for_torch = True, seed = None, veux_coin = False, lhs = False):
    #set up de la seed si voulue
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        rng.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if lhs:
        if nech > 5 and veux_coin:
            nech -=4
            init_coin = True
        else:
            init_coin = False
            
        argtir = [] #initialisation de la liste des parametre
        alpha, beta, N = calc_param(nech, d, nrechlhs, Nmin, Nmax)
    
        if all_opt:
            for i in range(nech):
                argtir.append([alpha[i], beta[i], Nmax]) #mise en forme de nos information
        
        else:
            N = np.random.randint(Nmin, (Nmax+1), nech)
            for i in range(nech):
                argtir.append([alpha[i], beta[i], N]) #mise en forme de nos information
    
        ech = constru_init(argtir, a, b, init_coin, Nmax, d)
        
        if init_coin:
            nech +=4
    else:
        ech =  [(torch.rand((Nmax,d), requires_grad=True, device=device)) for _ in range(nech)]

    if aff:
        affichage.aff_init(ech, a,b, 25, 5, 5, titre = "nuage_ini")
            
    if for_torch:
        for i in ech:
            i.requires_grad_(True)
    return ech



def initialisation(Nmin, Nmax, nech,jln_mth = False, a = torch.tensor([0,0]), b = torch.tensor([1,1]), d= 2,nrechlhs = 60000,
                 all_opt = True, aff = True, for_torch = True, seed = None, veux_coin = False, lhs = False):
    if jln_mth:
        return crea_ech_ini(Nmin, Nmax, nech, a, b, d,nrechlhs,
                         all_opt, aff, for_torch, seed, veux_coin, lhs), []
    else:
        ncpu = np.max([int(nech + 5),50])
        echcpu = rng.uniform(0,1,(ncpu,d))
        echdistpr = outils.dist2cp(echcpu, echcpu)
        echdist = []
        for i in range(ncpu-1):
            for j in range(i+1, ncpu):
                echdist.append(echdistpr[i,j])
        echdist = np.array(echdist).squeeze()
        echdist = np.sort(echdist)
        echdist = ((echdist/d))
        echdist = torch.tensor(echdist, dtype=torch.get_default_dtype())
        return crea_ech_ini(Nmin, Nmax, nech, a, b, d,nrechlhs,
                         all_opt, aff, for_torch, seed, veux_coin, lhs), echdist
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
