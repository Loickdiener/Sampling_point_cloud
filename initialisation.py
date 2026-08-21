import numpy as np
import torch
import numpy.random as rng
import outils
from geomloss import SamplesLoss
import random
import affichage



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device utilisé :", device)
torch.set_default_device(device)


sinkhorn = SamplesLoss(
    loss="sinkhorn",
    p=2,
    blur= 0.15
)


def LHS(nb_rechlhs, nb_sample, d):
    """
    Génère un plan d'échantillonnage Latin Hypercube (LHS) de type maximin.

    La fonction réalise plusieurs tirages aléatoires de configurations LHS et
    conserve celle dont la distance minimale entre deux points est la plus
    grande. Cette stratégie vise à maximiser la dispersion des points dans l'espace

    Args:
        nb_rechlhs (int): Nombre de configurations LHS candidates à générer et
            à évaluer.
        nb_sample (int): Nombre de points du plan d'échantillonnage.
        d (int): Dimension du problème. Le plan est construit sur ``2 * d``
            dimensions.

    Returns:
        list[np.ndarray]: Liste de permutations correspondant à la meilleure
        configuration LHS trouvée selon le critère maximin.
    """    
    maximindist = -0.1 #initialisation de notre maximum de disteance minimum
    bestperme = []  #initialisation de notre liste qui contient notre meilleur paterne
    for i in range(nb_rechlhs):
        
        permtemp = outils.lhs(nb_sample, 2*d)   #tirrage d'une configuration LHS
        C = outils.dist(np.array(permtemp).T, np.array(permtemp).T)    #Calcule de la matrice de distance
        np.fill_diagonal(C, 2**20) #On cherche la distance minimum donc afin de ne pas prendre les 0 de la diagonal on met une grande valeur à celle ci
        mintemp = np.min(C) # recherche de la distance minimal
        if mintemp > maximindist: #on enregistre notre trouvaille si il est mieux que tout ce que l'on a trouver jusqu'a present
            bestperme = permtemp
            maximindist = mintemp
    return bestperme


def calc_param(nb_sample, d, nb_rechlhs, Nmin, Nmax):
    """
    Génère les paramètres de lois bêta ainsi qu'un nombre de points associé à
    chaque échantillon à l'aide d'un plan d'expérience Latin Hypercube.

    Args:
        nb_sample (int): Nombre d'échantillons à générer.
        d (int): Nombre de dimensions du problème.
        nb_rechlhs (int): Nombre de tirages LHS candidats utilisés pour
            rechercher un plan de type maximin.
        Nmin (int): Nombre minimal de points à associer à un échantillon.
        Nmax (int): Nombre maximal de points à associer à un échantillon.

    Returns:
        tuple:
            - **alpha** (*np.ndarray*) : tableau de forme ``(nb_sample, d)``
              contenant les paramètres α des lois bêta ;
            - **beta** (*np.ndarray*) : tableau de forme ``(nb_sample, d)``
              contenant les paramètres β des lois bêta ;
            - **N** (*np.ndarray*) : vecteur de taille ``nb_sample``
              contenant le nombre de points associé à chaque échantillon.
    """
    esp = np.zeros((nb_sample,d))
    var = np.zeros((nb_sample,d))
    permu = LHS(nb_rechlhs, nb_sample, d)
    rangesp = np.linspace(0, 1, nb_sample + 1)
    for i in range(d):
        esp[:,i] = rng.uniform(rangesp[:-1], rangesp[1:])[permu[0]]
        rangevar = np.array(np.linspace(np.zeros(nb_sample), esp[:,i] * (1 - esp[:,i]),nb_sample + 1)).T
        for j in range(nb_sample):
            var[j,i] = rng.uniform(rangevar[j][permu[-d + i][j]], rangevar[j][(permu[-d+i][j] + 1)])
    #Calcule de notre alpha et beta grace à notre esp et var tirer precedement


    N = np.random.randint(Nmin, (Nmax+1), nb_sample)
    alpha = (np.array(esp) * (np.array(esp) *(1 - np.array(esp))-var))/(var)
    beta = ((1 - np.array(esp)) * (np.array(esp) *(1 - np.array(esp))-var))/(var)
    return alpha, beta, N





def constru_init(param_beta, born_inf, born_sup, init_coin, Nmax, d):
    """
    Construit un ensemble de configurations initiales pour une procédure
    d'optimisation ou d'exploration.

    Les configurations sont générées à partir de distributions bêta définies
    par les paramètres contenus dans ``param_beta``. Chaque ensemble de points
    est ensuite projeté sur le domaine borné par ``born_inf`` et ``born_sup``.

    Args:
        param_beta (Iterable[tuple]): Collection de triplets
            ``(alpha, beta, n_samples)`` où :
            - ``alpha`` contient les paramètres α des lois bêta ;
            - ``beta`` contient les paramètres β des lois bêta ;
            - ``n_samples`` est le nombre de points à générer.
        born_inf (torch.Tensor): Borne inférieure du domaine de définition,
            de forme ``(d,)``.
        born_sup (torch.Tensor): Borne supérieure du domaine de définition,
            de forme ``(d,)``.
        init_coin (bool): Si ``True``, ajoute des configurations initiales
            concentrées dans les coins du domaine. Utile uniquement pour d=2
        Nmax (int): Nombre de points utilisés pour chaque configuration
            générée dans les coins.
        d (int): Dimension de l'espace de travail.

    Returns:
        list[torch.Tensor]: Liste de nuages de points. Chaque élément de la
        liste est un tensor de forme ``(n_samples, d)`` ou ``(Nmax, d)``
        pour les configurations de coin.
    
    """
    ech = []
    
    if init_coin:
        A = torch.rand((Nmax,d))/4
        A += born_inf
        ech.append(A)
        A = torch.rand((Nmax,d))/4
        A[:,0] += born_inf[0]
        A[:,1] = born_sup[1] - A[:,1] 
        ech.append(A)
        A = torch.rand((Nmax,d))/4
        A[:,0] = born_sup[0] - A[:,0]
        A[:,1] += born_inf[1]
        ech.append(A)
        A = torch.rand((Nmax,d))/4
        A = born_sup - A
        ech.append(A)

    #tirage de nos echantillon
    for alpha, beta, n_samples in param_beta:
       # alpha, beta : (d,)
        alpha_t = torch.tensor(alpha)
        beta_t  = torch.tensor(beta)
        # Distribution Beta vectorisée
        dist = torch.distributions.Beta(alpha_t, beta_t)
        # Échantillonnage : (n_samples, d)
        samples = dist.sample((n_samples,))
        # Si tu veux le support [born_inf, b]
        samples = born_inf + (born_sup - born_inf) * samples
        ech.append(samples)
    return ech


    
#Fonction principale de génération des conditions initiales pour les algorithmes d'optimisation de nuages de points.
def crea_ech_ini(Nmin, Nmax, nb_sample, born_inf = torch.tensor([0,0]), born_sup = torch.tensor([1,1]), d= 2,nb_rechlhs = 60000,
                 all_opt = True, aff = True, for_torch = True, veux_coin = False, lhs = False):
    """
    Génère un ensemble de nuages de points servant d'initialisations pour une
    procédure d'optimisation.

    Selon les options choisies, les configurations initiales sont produites :

    - soit par un échantillonnage basé sur des lois bêta dont les paramètres
      sont générés via un plan Latin Hypercube ;
    - soit par des tirages uniformes aléatoires dans le domaine considéré.

    La fonction permet également de fixer une graine aléatoire pour assurer
    la reproductibilité des résultats, d'ajouter des configurations situées
    près des coins du domaine et d'afficher les nuages générés.

    Args:
        Nmin (int): Nombre minimal de points dans une configuration lorsque
            ``all_opt=False``.
        Nmax (int): Nombre maximal de points générés par configuration.
        nb_sample (int): Nombre de configurations initiales à produire.
        born_inf (torch.Tensor, optional): Borne inférieure du domaine de
            définition. Défaut : ``torch.tensor([0, 0])``.
        born_sup (torch.Tensor, optional): Borne supérieure du domaine de
            définition. Défaut : ``torch.tensor([1, 1])``.
        d (int, optional): Dimension de l'espace de travail. Défaut : ``2``.
        nb_rechlhs (int, optional): Nombre de configurations candidates
            explorées lors de la recherche du plan LHS de type maximin.
            Défaut : ``60000``.
        all_opt (bool, optional): Si ``True``, toutes les configurations sont
            générées avec ``Nmax`` points. Sinon, le nombre de points est tiré
            aléatoirement entre ``Nmin`` et ``Nmax``. Défaut : ``True``.
        aff (bool, optional): Si ``True``, affiche les configurations
            générées. Défaut : ``True``.
        for_torch (bool, optional): Si ``True``, active le calcul des gradients
            sur les nuages générés. Défaut : ``True``.
        veux_coin (bool, optional): Si ``True``, ajoute des configurations
            situées dans les coins du domaine lorsque le mode LHS est actif.
            Défaut : ``False``.
        lhs (bool, optional): Si ``True``, utilise l'initialisation basée sur
            les lois bêta et le plan Latin Hypercube. Sinon, génère des
            nuages uniformes aléatoires. Défaut : ``False``.

    Returns:
        list[torch.Tensor]: Liste de nuages de points. Chaque élément est un
        tensor de forme ``(N, d)`` représentant une configuration initiale.

    Notes:
        Si ``for_torch=True``, tous les tenseurs retournés possèdent
        ``requires_grad=True`` afin de permettre leur optimisation par
        descente de gradient.
    """
    
    if lhs:
        if nb_sample > 5 and veux_coin:
            nb_sample -=4
            init_coin = True
        else:
            init_coin = False
            
        argtir = [] #initialisation de la liste des parametre
        alpha, beta, N = calc_param(nb_sample, d, nb_rechlhs, Nmin, Nmax)
    
        if all_opt:
            for i in range(nb_sample):
                argtir.append([alpha[i], beta[i], Nmax]) #mise en forme de nos information
        
        else:
            N = np.random.randint(Nmin, (Nmax+1), nb_sample)
            for i in range(nb_sample):
                argtir.append([alpha[i], beta[i], N[i]]) #mise en forme de nos information
    
        ech = constru_init(argtir, born_inf, born_sup, init_coin, Nmax, d)
        
        if init_coin:
            nb_sample +=4
    else:
        ech =  [(torch.rand((Nmax,d), requires_grad=True, device=device)) for _ in range(nb_sample)]
        
    if for_torch:
        for i in ech:
            i.requires_grad_(True)
            
            
    if aff:
        affichage.aff_init(ech, born_inf,born_sup, d, 25, 5, 5, titre = "nuage_ini")
            
    return ech


    

def initialisation(Nmin, Nmax, nb_sample,jln_mth = False, born_inf = torch.tensor([0,0]), born_sup = torch.tensor([1,1]), d= 2,nb_rechlhs = 60000,
                 all_opt = True, aff = True, for_torch = True, seed = None, veux_coin = False, lhs = False):
    """
    Génère un ensemble de configurations initiales ainsi que, en option,
    une distribution de distances de référence.

    Cette fonction constitue le point d'entrée principal de l'initialisation
    des nuages de points. Elle s'appuie sur :func:`crea_ech_ini` pour produire
    les configurations candidates

    Args:
        Nmin (int): Nombre minimal de points dans une configuration lorsque
            ``all_opt=False``.
        Nmax (int): Nombre maximal de points par configuration.
        nb_sample (int): Nombre de configurations initiales à générer.
        jln_mth (bool, optional): Si ``True``, calcule également une
            distribution de distances de référence utilisée par la méthode
            jln(a utiliser que pour optimiser seulon cette 
            méthode). Défaut : ``False``.
        born_inf (torch.Tensor, optional): Borne inférieure du domaine.
            Défaut : ``torch.tensor([0, 0])``.
        born_sup (torch.Tensor, optional): Borne supérieure du domaine.
            Défaut : ``torch.tensor([1, 1])``.
        d (int, optional): Dimension de l'espace de travail.
            Défaut : ``2``.
        nb_rechlhs (int, optional): Nombre de tirages LHS candidats utilisés
            lors de la recherche d'un plan maximin. Défaut : ``60000``.
        all_opt (bool, optional): Si ``True``, toutes les configurations sont
            générées avec ``Nmax`` points. Défaut : ``True``.F
        aff (bool, optional): Si ``True``, affiche les configurations
            générées. Défaut : ``True``.
        for_torch (bool, optional): Si ``True``, active le calcul de gradient
            sur les tenseurs retournés. Défaut : ``True``.
        seed (int, optional): Graine aléatoire utilisée pour la
            reproductibilité des expériences.
        veux_coin (bool, optional): Si ``True``, ajoute des configurations
            concentrées dans les coins du domaine lorsque le mode LHS est
            actif. Défaut : ``False``.
        lhs (bool, optional): Si ``True``, utilise l'initialisation basée sur
            les lois bêta et le Latin Hypercube Sampling. Défaut : ``False``.

    Returns:
        tuple:
            - **ech** (*list[torch.Tensor]*) : liste des configurations
              initiales générées ;
            - **echdist** (*torch.Tensor | list*) :
                - liste vide lorsque ``jln_mth=False`` ;
                - distribution triée des distances quadratiques entre paires
                  de points d'un nuage uniforme de référence lorsque
                  ``jln_mth=True``.
    """
    #set up de la seed si voulue
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        rng.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        
    if not jln_mth:
        return crea_ech_ini(Nmin, Nmax, nb_sample, born_inf, born_sup, d,nb_rechlhs,
                         all_opt, aff, for_torch, veux_coin, lhs), []
    else:
        P_list = crea_ech_ini(Nmin, Nmax, nb_sample, born_inf, born_sup, d,nb_rechlhs,
                         all_opt, aff, for_torch, veux_coin, lhs)
        ncpu = np.max([int(nb_sample + 5),50])
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
        return P_list, echdist
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
