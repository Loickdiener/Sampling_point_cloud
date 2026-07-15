import numpy as np
import torch
import numpy.random as rng
from geomloss import SamplesLoss

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
    """
    Calcule une pénalité de répulsion entre les points d'un ensemble.

    Cette fonction mesure la proximité entre toutes les paires de points de
    ``P`` et applique une pénalité aux distances inférieures au seuil
    ``tol``. Plus les points sont proches les uns des autres, plus la
    pénalité est élevée. Les distances supérieures ou égales à ``tol`` ne
    contribuent pas à la pénalité.

    Args:
        P (torch.Tensor): Tensor de forme ``(n_points, n_dimensions)``
            contenant les coordonnées des points.
        tol (float, optional): Distance minimale souhaitée entre deux points.
            Les distances inférieures à cette valeur sont pénalisées.
            Défaut : ``1e-2``.

    Returns:
        torch.Tensor: Valeur scalaire correspondant à la pénalité moyenne
        appliquée sur l'ensemble des paires de points distinctes.
    """
    dist = torch.cdist(P, P)  # (n, n)
    mask = ~torch.eye(dist.size(0), dtype=bool, device=P.device)
    return (torch.relu(1*(tol - dist[mask]))).mean()



#Un tirage de configuration LHD/LHS
def lhs(n_points, n_dim):
    """
    Génère les permutations nécessaires à un plan d'échantillonnage de type
    Latin Hypercube Sampling (LHS).

    Args:
        n_points (int): Nombre de points à générer dans le plan
            d'échantillonnage.
        n_dim (int): Nombre de dimensions de l'espace de conception.

    Returns:
        list[np.ndarray]: Liste de longueur ``n_dim`` contenant, pour chaque
        dimension, une permutation aléatoire des entiers de ``0`` à
        ``n_points - 1``.

    Notes:
        Cette fonction ne génère pas directement les coordonnées du plan LHS,
        mais uniquement les permutations utilisées pour sa construction.
    """
    perm = []
    for j in range(n_dim):
        perm.append(rng.permutation(n_points))
        
    return perm

def dist(X, Y, len_x=None, len_y=None):
    """
    Calcule la matrice des distances euclidiennes au carré entre deux ensembles 
    de points.
    
    Args:
        X (np.ndarray): Tableau de forme ``(n_x, n_dim)`` contenant le premier
            ensemble de points.
        Y (np.ndarray): Tableau de forme ``(n_y, n_dim)`` contenant le second
            ensemble de points.
        len_x (int, optional): Nombre de points de ``X`` à considérer. Si non
            renseigné, utilise ``len(X)``.
        len_y (int, optional): Nombre de points de ``Y`` à considérer. Si non
            renseigné, utilise ``len(Y)``.

    Returns:
        np.ndarray: Matrice de forme ``(lx, ly)`` dont chaque élément est la
        distance euclidienne au carré entre une paire de points de ``X`` et
        ``Y``.
    """
    if(len_x == None):
        len_x = len(X)
    if(len_y == None):  
        len_y = len(Y)
    C = np.zeros([len_x,len_y])
    C = np.sum((X[:, None, :] - Y[None, :, :])**2, axis=2)
    return C

def borne(Y, born_inf, born_sup, d=2, eps=1e-6):
    """
    Calcule une pénalité logarithmique de proximité aux bornes d'un domaine.

    Args:
        Y (torch.Tensor): Tensor contenant les points à contraindre. La
            dernière dimension doit être de taille ``d``.
        born_inf (torch.Tensor ou array-like): Bornes inférieures du domaine. Si un
            unique scalaire est fourni, il est répliqué sur les ``d``
            dimensions.
        born_sup (torch.Tensor ou array-like): Bornes supérieures du domaine. Si un
            unique scalaire est fourni, il est répliqué sur les ``d``
            dimensions.
        d (int, optional): Nombre de dimensions du problème. Défaut : ``2``.
        eps (float, optional): Constante ajoutée aux arguments des logarithmes
            pour éviter les instabilités numériques. Défaut : ``1e-6``.

    Returns:
        torch.Tensor: Valeur scalaire correspondant à la pénalité moyenne
        associée au respect des bornes.

    Notes:
        Plus un point est proche d'une borne, plus la pénalité augmente.
        Les points situés en dehors du domaine peuvent produire des valeurs
        non définies (NaN) en raison du logarithme.
    """
    if len(born_inf) != d:
        if len(born_inf) == 1:
            born_inf = torch.ones(d)*born_inf
        else:
            born_inf = torch.zeros(d)
            
    if len(born_sup) != d:
        if len(born_sup) == 1:
            born_sup = torch.ones(d)*born_sup
        else:
            born_sup = torch.zeros(d)
    return -torch.mean(torch.log(Y - born_inf + eps) + torch.log(born_sup - Y + eps))


def bornew(Y, eps=1e-6):
    """
    Calcule une pénalité logarithmique de proximité aux bornes de l'intervalle
    [0, 1].

    Args:
        Y (torch.Tensor): Tensor contenant les valeurs à contraindre dans
            l'intervalle ``[0, 1]``.
        eps (float, optional): Constante ajoutée aux arguments des logarithmes
            pour éviter les instabilités numériques et les logarithmes de
            zéro. Défaut : ``1e-6``.

    Returns:
        torch.Tensor: Valeur scalaire correspondant à la pénalité moyenne
        associée à la proximité des bornes 0 et 1.
    """
    return -torch.mean( torch.log(Y + eps) +   torch.log(1.0 - Y + eps))


def min_dist(X, Y, weight_Y):
    """
    Calcule la distance de transport optimale minimale entre un ensemble de
    points et une distribution de référence.

    Pour chaque point de ``X``, cette fonction évalue sa distance de transport
    optimale (Sinkhorn) à la mesure discrète définie par les points ``Y`` et
    leurs masses ``mY``. La plus petite de ces distances est ensuite
    retournée.

    Args:
        X (torch.Tensor): Tensor de forme ``(n_x, d)`` contenant les points
            candidats.
        Y (torch.Tensor): Tensor de forme ``(n_y, d)`` définissant le support
            de la distribution de référence.
        weight_Y (torch.Tensor): Tensor de forme ``(n_y,)`` contenant les masses
            associées aux points de ``Y``. Les masses sont supposées former
            une distribution de probabilité.

    Returns:
        torch.Tensor: Valeur scalaire correspondant à la plus petite distance
        de Sinkhorn observée entre un point de ``X`` et la distribution de
        référence ``(Y, mY)``.

   Notes:
        Une faible valeur indique qu'au moins un point de ``X`` est proche,
        au sens du transport optimal régularisé, de la distribution de
        référence.

    """
    vals = []
    for Xid in X:
        mXid = torch.ones(1, device=X.device)  # masse uniforme
        vals.append(sinkhorn(mXid, Xid.unsqueeze(0), weight_Y, Y))
    return torch.min(torch.stack(vals))

def gaussian_kernel(X, Y, sigma=.5):
    """
    Calcule la matrice du noyau gaussien (RBF) entre deux ensembles de points.
    
    Args:
        X (torch.Tensor): Tensor de forme ``(n_x, d)`` contenant le premier
            ensemble de points.
        Y (torch.Tensor): Tensor de forme ``(n_y, d)`` contenant le second
            ensemble de points.
        sigma (float, optional): Paramètre de largeur du noyau gaussien.
            Plus ``sigma`` est grand, plus la décroissance avec la distance est
            lente. Défaut : ``0.5``.

    Returns:
        torch.Tensor: Matrice de forme ``(n_x, n_y)`` dont l'élément
        ``(i, j)`` représente la similarité gaussienne entre ``X[i]`` et
        ``Y[j]``.
    """   
    D2 = torch.cdist(X, Y, p=2) **2
    return torch.exp(-D2 / (2 * sigma ** 2))



def MMD_n(X, Y, kern=gaussian_kernel, sigma=.5, Kyy=None, Kxx=None):
    """
    Calcule la Maximum Mean Discrepancy (MMD) empirique entre deux ensembles
    de points à l'aide d'un noyau reproduisant.

    Args:
        X (torch.Tensor): Tensor de forme ``(n_x, d)`` contenant le premier
            ensemble d'échantillons.
        Y (torch.Tensor): Tensor de forme ``(n_y, d)`` contenant le second
            ensemble d'échantillons.
        kern (callable, optional): Fonction de noyau utilisée pour calculer
            les matrices de Gram. La fonction doit accepter les arguments
            ``(X, Y, sigma)``. Défaut : ``gaussian_kernel``.
        sigma (float, optional): Paramètre de largeur du noyau. Défaut :
            ``0.5``.
        Kyy (torch.Tensor, optional): Matrice de Gram pré-calculée pour
            ``Y``. Si fournie, son calcul est évité.
        Kxx (torch.Tensor, optional): Matrice de Gram pré-calculée pour
            ``X``. Si fournie, son calcul est évité.

    Returns:
        torch.Tensor: Valeur scalaire correspondant à la MMD empirique entre
        les ensembles ``X`` et ``Y``.
    """
    # Matrices de noyau
    if Kxx is None:
        Kxx = kern(X, X, sigma)/len(X)**2
    if Kyy is None:
        Kyy = kern(Y, Y, sigma)/len(Y)**2
    Kxy = kern(X, Y, sigma)/(len(X)*len(Y))
    return (Kxx.sum() + Kyy.sum() - 2 * Kxy.sum())



def MMD_p(X, weight_X, Y, weight_Y, kern=gaussian_kernel, sigma=.5, Kyy=None, Kxx=None):
    """
    Calcule la Maximum Mean Discrepancy (MMD) pondérée entre deux mesures
    discrètes à l'aide d'un noyau reproduisant.

    Args:
        X (torch.Tensor): Tensor de forme ``(n_x, d)`` contenant les points de
            la première mesure.
        weight_X (torch.Tensor): Tensor de forme ``(n_x,)`` contenant les poids
            associés aux points de ``X``.
        Y (torch.Tensor): Tensor de forme ``(n_y, d)`` contenant les points de
            la seconde mesure.
        weight_Y (torch.Tensor): Tensor de forme ``(n_y,)`` contenant les poids
            associés aux points de ``Y``.
        kern (callable, optional): Fonction de noyau utilisée pour calculer
            les matrices de Gram. Elle doit accepter les arguments
            ``(X, Y, sigma)``. Défaut : ``gaussian_kernel``.
        sigma (float, optional): Paramètre de largeur du noyau. Défaut :
            ``0.5``.
        Kyy (torch.Tensor, optional): Matrice de Gram pondérée pré-calculée
            pour ``Y``. Si fournie, son calcul est évité.
        Kxx (torch.Tensor, optional): Matrice de Gram pondérée pré-calculée
            pour ``X``. Si fournie, son calcul est évité.

    Returns:
        torch.Tensor: Valeur scalaire correspondant à la MMD pondérée entre
        les mesures discrètes définies par ``(X, Wx)`` et ``(Y, Wy)``.
    """
    # Matrices de noyau
    if Kxx is None:
        Kxx = weight_X[:, None] * weight_X[None,:] *kern(X, X, sigma)
    if Kyy is None:
        Kyy = weight_Y[:, None] * weight_Y[None,:]*  kern(Y, Y, sigma)
    Kxy = weight_X[:, None] * weight_Y[None,:] * kern(X, Y, sigma)
    return (Kxx.sum() + Kyy.sum() - 2 * Kxy.sum())


def min_MMD_p(ech, W_ech, Y, W, kern, sigma=.5):
    """
    Calcule la plus petite valeur de MMD pondérée entre plusieurs mesures
    candidates et une mesure de référence.
    
    Args:
        ech (Iterable[torch.Tensor]): Collection de supports discrets
            candidats. Chaque élément est un tensor de forme ``(n_i, d)``
            contenant les points de la mesure candidate.
        W_ech (Iterable[torch.Tensor]): Collection des vecteurs de poids
            associés aux éléments de ``ech``. Chaque vecteur est de forme
            ``(n_i,)``.
        Y (torch.Tensor): Tensor de forme ``(n_y, d)`` contenant le support de
            la mesure de référence.
        W (torch.Tensor): Tensor de forme ``(n_y,)`` contenant les poids de la
            mesure de référence.
        kern (callable): Fonction de noyau utilisée pour le calcul de la MMD.
            Elle doit accepter les arguments ``(X, Y, sigma)``.
        sigma (float, optional): Paramètre du noyau. Défaut : ``0.5``.

    Returns:
        torch.Tensor: Valeur scalaire correspondant à la plus petite MMD²
        entre les mesures candidates et la mesure de référence.
    """
    Kyy = W[:, None] * W[None,:].T * kern(Y, Y, sigma)
    return torch.min(torch.stack([
        MMD_p(e, we, Y, W, kern, sigma, Kyy) for e, we in zip(ech, W_ech)
    ]))


def get_masked_points(P, w):
    """
    Convertit un vecteur de scores en poids de sélection via une fonction
    sigmoïde.

    Args:
        P (torch.Tensor): Tensor contenant les points ou échantillons à
            considérer. Ce tensor est retourné inchangé.
        w (torch.Tensor): Tensor de scores réels associé aux points de ``P``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - ``P`` : les points d'entrée, inchangés ;
            - ``m`` : le masque continu obtenu par application de la fonction
              sigmoïde à ``w``.
    """
    m = torch.sigmoid(w)
    return P, m
 
def quasimin(x, tau=0.1):
    """
    Calcule une approximation différentiable du minimum d'un ensemble de
    valeurs à l'aide de l'opérateur softmin.

    Args:
        x (torch.Tensor): Tensor contenant les valeurs dont on souhaite
            approximer le minimum.
        tau (float, optional): Paramètre de température contrôlant le degré
            de lissage. Une valeur faible produit une approximation plus
            proche du minimum exact. Défaut : ``0.1``.

    Returns:
        torch.Tensor: Approximation différentiable du minimum des valeurs
        contenues dans ``x``.
    """
    return -tau * torch.logsumexp(-x / tau, dim=0)



def distsm(nb_point):
    """
    Calcule la somme des distances quadratiques entre toutes les paires de
    points d'un ensemble.
    
    Args:
        nb_point (Iterable[torch.Tensor]): Collection de points. Chaque point
            doit être représenté par un tensor de même dimension.

    Returns:
        torch.Tensor: Somme des distances euclidiennes au carré entre toutes
        les paires non ordonnées de points.
    """
    return torch.sum(torch.stack([(x - y)**2 for x in nb_point for y in nb_point]))/2


def soft_count_near(x, c=0.0, temperature=10.0):
    """
    Calcule un comptage souple (différentiable) des valeurs proches d'une
    cible donnée.
    
    Args:
        x (torch.Tensor): Tensor contenant les valeurs à évaluer.
        c (float, optional): Valeur cible autour de laquelle le comptage est
            effectué. Défaut : ``0.0``.
        temperature (float, optional): Paramètre contrôlant la largeur de la
            zone d'influence autour de ``c``. Plus cette valeur est grande,
            plus seules les valeurs très proches de ``c`` contribuent au
            résultat. Défaut : ``10.0``.

    Returns:
        torch.Tensor: Valeur scalaire représentant le comptage souple des
        éléments proches de ``c``.
    """
    return torch.exp(-temperature * (x - c)**2).sum()

def w_penal(w):
    """
    Calcule un terme favorisant des poids de grande amplitude.

    Args:
        w (torch.Tensor): Tensor contenant les poids ou paramètres à
            régulariser.

    Returns:
        torch.Tensor: Valeur scalaire correspondant à la somme des termes
        ``exp(-w²)``.
    """
    return torch.exp(-w**2).sum()


def uniform_test_stat(inertias, born_inf=0., born_sup=nech**(-1/d)):
    """
    Calcule une statistique de type Kolmogorov-Smirnov mesurant l'écart entre
    la distribution empirique des inerties et une loi uniforme.

    Args:
        inertias (torch.Tensor): Tensor unidimensionnel contenant les valeurs
            d'inertie à analyser.
        born_inf (float, optional): Borne inférieure utilisée pour la
            normalisation. Si les données contiennent une valeur plus petite,
            cette dernière est utilisée à la place. Défaut : ``0.``.
        born_sup (float, optional): Borne supérieure utilisée pour la
            normalisation. Si les données contiennent une valeur plus grande,
            cette dernière est utilisée à la place. Défaut :
            ``nech**(-1/d)``.

    Returns:
        torch.Tensor: Valeur scalaire correspondant à la statistique de
        Kolmogorov-Smirnov.
    """
    with torch.no_grad():
        born_inf = torch.min(torch.min(inertias), torch.tensor(born_inf))
        born_sup = torch.max(torch.max(inertias), torch.tensor(born_sup))
    u = (inertias - born_inf) / (born_sup - born_inf)
    u_sorted, _ = torch.sort(u)

    n = u.shape[0]
    i = torch.arange(1, n + 1, device=u.device)

    # ECDF vs uniforme
    D_plus  = torch.max(i / n - u_sorted)
    D_minus = torch.max(u_sorted - (i - 1) / n)

    return torch.max(D_plus, D_minus)



def cvm_uniform_loss(inertias, born_inf=0, born_sup=1.2 * Nmax**(-1/d)):
    """
    Calcule une perte de type Cramér-von Mises mesurant l'écart entre la
    distribution des valeur et une loi uniforme.

    Args:
        inertias (torch.Tensor): Tensor unidimensionnel contenant les valeurs
            à analyser.
        born_inf (float, optional): Borne inférieure utilisée pour la
            normalisation. Défaut : ``0``.
        born_sup (float, optional): Borne supérieure utilisée pour la
            normalisation. Si elle est inférieure à la plus grande inertie
            observée, elle est automatiquement augmentée. Défaut :
            ``1.2 * Nmax**(-1/d)``.

    Returns:
        tuple:
            - **loss** (*torch.Tensor*) : perte de type Cramér-von Mises ;
            - **born_inf** (*float ou torch.Tensor*) : borne inférieure utilisée ;
            - **born_sup** (*float ou torch.Tensor*) : borne supérieure finale
              utilisée pour la normalisation.
    """
    if torch.tensor(born_sup)<torch.max(inertias):
        born_sup= torch.max(inertias) + (born_sup-born_inf)*.01
    u = (inertias - born_inf) / (born_sup - born_inf)
    u_sorted, _ = torch.sort(u)

    n = u.shape[0]
    i = torch.arange(1, n + 1, device=u.device, dtype=u.dtype)

    target = (2*i - 1) / (2*n)

    return torch.mean((u_sorted - target)**2), born_inf, born_sup


def test_dist_a_unif(inertias, born_inf=0., born_sup=1.2 * Nmax**(-1/d)):
    """
    Évalue la proximité entre la distribution des inerties et une distribution
    uniforme à l'aide d'une distance de transport optimal régularisée.

    Args:
        inertias (torch.Tensor): Tensor unidimensionnel contenant les valeurs
            d'inertie à analyser.
        born_inf (float, optional): Borne inférieure de l'intervalle de
            référence. Si elle est supérieure à la plus petite inertie
            observée, elle est automatiquement ajustée. Défaut : ``0.``.
        born_sup (float, optional): Borne supérieure de l'intervalle de
            référence. Si elle est inférieure à la plus grande inertie
            observée, elle est automatiquement ajustée. Défaut :
            ``1.2 * Nmax**(-1/d)``.

    Returns:
        tuple:
            - **val** (*torch.Tensor*) : distance de Sinkhorn entre les
              inerties observées et un échantillon uniforme de référence ;
            - **born_inf** (*float ou torch.Tensor*) : borne inférieure
              effectivement utilisée ;
            - **born_sup** (*float ou torch.Tensor*) : borne supérieure
              effectivement utilisée.
    """
    if torch.tensor(born_sup)<torch.max(inertias):
        born_sup= torch.max(inertias) + (born_sup-born_inf)*.01
    if torch.tensor(born_inf)> torch.min(inertias):
        born_inf = torch.min(inertias) - (born_sup-born_inf)*.0001
    ref = (torch.rand((len(inertias))) + born_inf) * (born_sup - born_inf)
    val = sinkhorn(inertias.view(-1,1),ref.view(-1,1))
    return val, born_inf, born_sup


def dist2cp(X, Y, lx=None, ly=None):
    """
    Calcule la matrice des distances euclidiennes au carré entre deux ensembles
    de points.

    Args:
        X (np.ndarray): Tableau de forme ``(n_x, d)`` contenant le premier
            ensemble de points.
        Y (np.ndarray): Tableau de forme ``(n_y, d)`` contenant le second
            ensemble de points.
        lx (int, optional): Nombre de points de ``X`` à considérer. Si non
            renseigné, utilise ``len(X)``.
        ly (int, optional): Nombre de points de ``Y`` à considérer. Si non
            renseigné, utilise ``len(Y)``.

    Returns:
        np.ndarray: Matrice de forme ``(lx, ly)`` dont l'élément ``(i, j)``
        contient la distance euclidienne au carré entre ``X[i]`` et ``Y[j]``.
    """
    if(lx == None):
        lx = len(X)
    if(ly == None):  
        ly = len(Y)
    C = np.zeros([lx,ly])
    C = np.sum((X[:, None, :] - Y[None, :, :])**2, axis=2)
    return C

def distcp(X, Y):
    """
    Calcule les distances quadratiques élément par élément entre deux ensembles
    de valeurs unidimensionnelles.

    Args:
        X (np.ndarray ou torch.Tensor): Vecteur de forme ``(n_x,)`` contenant
            le premier ensemble de valeurs.
        Y (np.ndarray ou torch.Tensor): Vecteur de forme ``(n_y,)`` contenant
            le second ensemble de valeurs.

    Returns:
        np.ndarray ou torch.Tensor: Matrice de forme ``(n_x, n_y)``
    """
    C = (X[:, None] - Y[None, :])**2
    return C

def kernel(x, y, sigma = .01):
    """Version unidimensionnelle de gaussian_kernel. Présent ci-dessus"""
    return torch.exp(- (x[:, None] - y[None, :])**2 / (2*sigma**2))



