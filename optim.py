import numpy as np
import matplotlib.pyplot as plt
import torch
import outils
from geomloss import SamplesLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def variable_changeante(it, Nmin, Nmax, peri_p, peri_w, mu0, amplificateur):
    """
    Met à jour les paramètres de contrôle de l'optimisation en fonction de
    l'itération courante.

    Cette fonction orchestre l'alternance entre l'optimisation des positions
    des points et celle des poids de sélection. Elle gère également les
    périodes dédiées à chaque type de variable, l'évolution d'un facteur
    d'amplification et la décroissance d'un coefficient de régularisation.

    Args:
        it (int): Numéro de l'itération courante.
        Nmin (int): Nombre minimal de points autorisés.
        Nmax (int): Nombre maximal de points autorisés.
        peri_p (int): Nombre d'itérations restantes durant lesquelles seules
            les positions doivent être optimisées.
        peri_w (int): Nombre d'itérations restantes durant lesquelles seuls
            les poids doivent être optimisés.
        mu0 (float): Valeur initiale du coefficient de régularisation.
        amplificateur (float): Valeur actuelle du facteur d'amplification.

    Returns:
        tuple:
            - **turn_p** (*bool*) : indique si l'itération doit être consacrée
              à l'optimisation des positions ;
            - **peri_p** (*int*) : compteur mis à jour de la période dédiée
              aux positions ;
            - **peri_w** (*int*) : compteur mis à jour de la période dédiée
              aux poids ;
            - **mu** (*float*) : coefficient de régularisation mis à jour ;
            - **amplificateur** (*float*) : facteur d'amplification mis à
              jour.

    Notes:
        La variable ``turn_p`` détermine si l'étape courante concerne les
        positions(turn_p = True) ou les poids(turn_p = False). Le schéma 
        d'alternance dépend du nombre d'itérations déjà effectuées.
    """
    turn_p = (((it % 5 != 0 and it < 300) or (it % 2 == 0 and it>300 and it<10000) or (it % 100 > 4 and it>10000)) or (Nmin==Nmax))
    if Nmin !=Nmax:
        if it>300 and peri_p ==0 and it % 200 == 0:
            peri_w = 10
        if it>300 and peri_w ==0 and it % 200 == 100:
            peri_p = 10
        
        if peri_w>0:
            turn_p = False
            peri_w -=1
        if peri_p>0:
            peri_p -=1
            turn_p = True
    
    if it % 50 == 0:
        amplificateur = min(1 + it/10 ,Nmax * 10)
        
    mu = max(1e-5, mu0 * (0.99 ** it))
    return turn_p, peri_p, peri_w, mu, amplificateur

def calc_point_poid_nb_inter(nb_samples, Nmax, d, cloud_list, weight_list, amplificateur, turn_p):
    """
    Prépare les nuages de points, les poids normalisés et les nombres de
    points effectifs utilisés lors d'une itération d'optimisation.

    Cette fonction applique la transformation de masquage aux poids de chaque
    nuage, calcule les poids normalisés associés et construit des tenseurs
    regroupant l'ensemble des nuages et des poids.

    Args:
        nb_samples (int): Nombre de nuages traités simultanément.
        Nmax (int): Nombre maximal de points par nuage.
        d (int): Dimension des points.
        cloud_list (list[torch.Tensor]): Liste des nuages de points. Chaque
            élément doit être un tenseur de forme ``(Nmax, d)`` et ou chaque points
            de chaque nuage doit egalement etre un tensor.
        weight_list (list[torch.Tensor]): Liste des vecteurs de poids associés aux
            nuages de ``cloud_list``.
        amplificateur (float): Facteur multiplicatif appliqué aux poids avant
            calcul du masque. Cette variable sert a avoir une selection des 
            point actif plus tranché.
        turn_p (bool): Indique si l'itération courante est dédiée à
            l'optimisation des positions des points.

    Returns:
        tuple:
            - **P_masked** (*torch.Tensor*) : tenseur de forme
              ``(nb_samples, Nmax, d)`` contenant les positions des points ;
            - **m_masked** (*torch.Tensor*) : tenseur de forme
              ``(nb_samples, Nmax)`` contenant les poids normalisés ;
            - **nb_points** (*torch.Tensor*) : vecteur contenant, pour chaque
              nuage, la somme des poids avant normalisation.
    """
    P_masked = torch.empty((nb_samples, Nmax, d), device=cloud_list[0].device)
    m_masked = torch.empty((nb_samples, Nmax), device=cloud_list[0].device)
    nb_points = torch.empty(nb_samples, device=cloud_list[0].device)

    # Préparation des nuages
    for i, (P, w) in enumerate(zip(cloud_list, weight_list)):
        Pi, mi = outils.get_masked_points(P, w * amplificateur)
        nb_points[i] = (mi.sum())
        mi = mi/(mi.sum())  
        if turn_p:
            P_masked[i] = Pi
            m_masked[i] = mi.detach()
        else:
            P_masked[i] = Pi.detach()
            m_masked[i] = mi
    return P_masked, m_masked, nb_points


def dist_min_par_ech(nech, cloud_list, P_masked, weight_list):
    """
    Calcule, pour chaque nuage de points, une approximation différentiable de
    la distance minimale entre deux points distincts.

    Cette fonction évalue les distances euclidiennes entre toutes les paires
    de points d'un même nuage puis utilise une approximation lisse du minimum
    afin d'obtenir une quantité compatible avec une optimisation par gradient.

    Args:
        nech (int): Nombre de nuages de points à traiter.
        cloud_list (list[torch.Tensor]): Liste des nuages de points. Utilisée
            uniquement pour récupérer le périphérique de calcul (CPU/GPU).
        P_masked (torch.Tensor): Tensor de forme ``(nech, Nmax, d)``
            contenant les coordonnées des points de chaque nuage.
        weight_list (torch.Tensor): Tensor de poids ``(nech, Nmax)``
            contenant les poids associer à chaque points de chaque nuage.)

    Returns:
        torch.Tensor: Vecteur de taille ``nech`` contenant, pour chaque
        nuage, une approximation de la distance minimale entre deux points
        distincts.
    """
    min_intra_dist = torch.empty(nech, device=cloud_list[0].device)
    
    for i in range(nech):
        dist_pour_intra = torch.cdist(P_masked[i], P_masked[i])
        neg = (weight_list[i] < 0).nonzero(as_tuple=True)[0]
        k, j = torch.triu_indices(dist_pour_intra.size()[0], dist_pour_intra.size()[0], offset=1)
        mask = (~torch.isin(k, neg)) & (~torch.isin(j, neg))
        k = k[mask]
        j = j[mask]
        intra_all_dist = dist_pour_intra[k,j]
        min_intra_dist[i] = outils.quasimin(intra_all_dist, 0.001)
    return min_intra_dist


def calc_min_dist_inter_nuage(nb_samples, P_masked, m_masked, Nmin, Nmax, it,
                              sinkhorn, ofr_comp, jln_mth, echdist=[], Wasserstein = True):
    """
    Calcule une mesure de séparation entre un ensemble de nuages de points.

    Cette fonction évalue une distance entre toutes les paires de nuages
    distincts puis retourne soit la plus petite distance obtenue, soit un
    score comparant la distribution des distances observées à une
    distribution de référence.

    Deux métriques de comparaison sont disponibles :

    - la distance de transport optimal régularisée (Sinkhorn) ;
    - la Maximum Mean Discrepancy (MMD) basée sur un noyau gaussien.

    Args:
        nb_samples (int): Nombre de nuages de points.
        P_masked (torch.Tensor): Tensor de forme
            ``(nb_samples, Nmax, d)`` contenant les positions des points.
        m_masked (torch.Tensor): Tensor de forme
            ``(nb_samples, Nmax)`` contenant les poids normalisés associés aux
            points.
        Nmin (int): Nombre minimal de points autorisé.
        Nmax (int): Nombre maximal de points autorisé.
        sinkhorn (callable): Fonction de calcul de la distance de Sinkhorn.
        ofr_comp (callable): Fonction utilisée pour comparer la distribution
            des distances observées à une distribution de référence.
        jln_mth (bool): Active la comparaison entre distributions de
            distances.
        echdist (torch.Tensor, optional): Distribution de distances de
            référence utilisée lorsque ``jln_mth=True``.
            Défaut : ``[]``.
        Wasserstein (bool, optional): Si ``True``, utilise la distance de
            Sinkhorn. Si ``False``, utilise une distance de type MMD basée
            sur un noyau gaussien. Défaut : ``True``.

    Returns:
        torch.Tensor:
            - si ``jln_mth=False`` : plus petite distance entre deux nuages 
                exact ou approché selon l'iteration d'appel;
            - si ``jln_mth=True`` : score de comparaison entre la
              distribution des distances calculées et ``echdist``.
    """
    i, j = torch.triu_indices(nb_samples, nb_samples, offset=1)
    
    P_i = P_masked[i]        # (num_pairs, M, d)
    P_j = P_masked[j]
    
    w_i = m_masked[i]         # (num_pairs, M)
    w_j = m_masked[j]
    # calcul Sinkhorn pairwise (batché)
    if  Wasserstein:
        if Nmin != Nmax:
            all_dist = sinkhorn(w_i, P_i, w_j, P_j)
        else:
            all_dist = sinkhorn(P_i, P_j)
    else:
        sigma = 0.5
        
        X_i = P_masked[:, None, :, :]  
        X_j = P_masked[None, :, :, :]

        D = torch.cdist(X_i, X_j)**2
        
        if Nmin != Nmax:
            m_i = m_masked[:, None, :, None] 
            m_j = m_masked[None, :, None, :]
            D_xx = torch.cdist(P_masked, P_masked)**2
            K_xx = m_masked[:, :, None] * m_masked[:, None, :] * torch.exp(-D_xx / (2 * sigma**2))
        else:
            m_i = (torch.ones((nb_samples,Nmax))/Nmax)[:, None, :, None] 
            m_j = (torch.ones((nb_samples,Nmax))/Nmax)[None, :, None, :]
            D_xx = torch.cdist(P_masked, P_masked)**2
            K_xx = torch.exp(-D_xx / (2 * sigma**2))/ Nmax**2

            
        K_xy = m_i * m_j * torch.exp(-D / (2 * sigma**2))
        sum_xy = K_xy.sum(dim=(-1, -2))  
        
        
        sum_xx = K_xx.sum(dim=(-1, -2))
        
        MMD = sum_xx[:, None] + sum_xx[None, :] - 2 * sum_xy
        
        i, j = torch.triu_indices(nb_samples, nb_samples, offset=1)
        all_dist = MMD[i, j]
    
    if jln_mth:
        return ofr_comp(all_dist.view(-1,1), echdist.view(-1,1))
    else:
        if it>(200):
            return torch.min(all_dist)
        else:
            return outils.quasimin(all_dist, .05)
        





def calc_loss_p(a, b, d, cloud_list, pds0, loss_dist, pena_repul,
                nb_samples, intra_repul_pena, mu0):
    """
    Calcule la fonction de coût utilisée lors de l'optimisation des positions
    des points.

    Cette fonction combine plusieurs termes de perte visant à produire des
    nuages de points bien répartis dans le domaine d'étude :

    - un terme favorisant l'éloignement des nuages entre eux ;
    - un terme de pénalisation lié à la distance minimal entre les points des nuages
    de points;
    - une barrière logarithmique empêchant les points de sortir du domaine ;
    - une pénalité de répulsion interne entre les points.

    Args:
        a (torch.Tensor): Borne inférieure du domaine de définition.
        b (torch.Tensor): Borne supérieure du domaine de définition.
        d (int): Dimension de l'espace.
        cloud_list (list[torch.Tensor]): Liste des nuages de points optimisés.
        pds0 (float): Coefficient associé au terme de distance entre nuages.
        loss_dist (torch.Tensor): Critère mesurant la séparation entre les
            nuages.
        pena_rep (torch.Tensor): Pénalité de répulsion entre les points.
        nb_samples (int): Nombre de nuages considérés.
        intra_repul_pena (torch.Tensor): Terme de pénalisation basé sur l'inertie ou
            la répartition des solutions.
        mu0 (float): Coefficient de régularisation associé à la contrainte de
            domaine.

    Returns:
        torch.Tensor: Valeur scalaire correspondant à la fonction de coût
        globale.
    """
    regp = torch.mean(torch.stack([outils.borne(P ,a ,b , d) for P in cloud_list]))
    return (- pds0 * loss_dist + nb_samples/5 * intra_repul_pena + mu0 * regp + pena_repul)


def calc_loss_w(Nmin, Nmax, nb_points, num_sizes, cloud_list, nb_pnt_possible,
                temp, weight_list, nb_samples, counter_w_opt, it, alpha, beta,
                gamma, delta, loss_dist, mu, pds0, pds1, pds2, pds3):
    """
    Calcule la fonction de coût utilisée lors de l'optimisation des poids de
    sélection des points.

    Cette fonction combine plusieurs termes visant à contrôler :
    - la séparation entre les nuages ;
    - la répartition des cardinalités obtenues ;
    - le respect des contraintes sur le nombre de points ;
    - la polarisation des poids de sélection.
    
    Les coefficients de pondération des différents termes sont ajustés
    périodiquement afin d'équilibrer leurs contributions respectives au cours
    de l'optimisation.

    Args:
        Nmin (int): Nombre minimal de points autorisé.
        Nmax (int): Nombre maximal de points autorisé.
        nb_points (torch.Tensor): Nombre effectif de points associé à chaque
            nuage.
        num_sizes (int): Nombre de cardinalités admissibles, généralement égal à
            ``Nmax - Nmin + 1``.
        cloud_list (list[torch.Tensor]): Liste des nuages de points.
        nb_pnt_possible (torch.Tensor ou array-like): Ensemble des nombres de
            points admissibles.
        temp (float): Paramètre utilisé pour le comptage différentiable des
            cardinalités.
        weight_list (list[torch.Tensor]): Liste des vecteurs de poids(dans R).
        nb_samples (int): Nombre de nuages optimisés simultanément.
        counter_w_opt (int): Compteur du nombre d'appels à la phase
            d'optimisation des poids.
        it (int): Numéro de l'itération courante.
        alpha (float): Poids relatif du critère principal.
        beta (float): Poids relatif du critère de répartition.
        gamma (float): Poids relatif de la régularisation du nombre de points.
        delta (float): Poids relatif de la régularisation des poids.
        loss_dist (torch.Tensor): Critère principal de séparation entre les
            nuages.
        mu (float): Valeur maximale autorisée pour le coefficient associé à
            la régularisation de taille.
        pds0 (float): Coefficient du critère principal.
        pds1 (float): Coefficient du critère de répartition.
        pds2 (float): Coefficient du critère de régularisation de taille.
        pds3 (float): Coefficient du critère de régularisation des poids.

    Returns:
        tuple:
            - **loss** (*torch.Tensor*) : valeur de la fonction de coût ;
            - **counter_w_opt** (*int*) : compteur mis à jour ;
            - **pds1** (*float ou torch.Tensor*) : coefficient actualisé de la
              pénalité de répartition ;
            - **pds2** (*float ou torch.Tensor*) : coefficient actualisé de la
              régularisation de taille ;
            - **pds3** (*float ou torch.Tensor*) : coefficient actualisé de la
              pénalité sur les poids.

    Notes:
        Le critère est composé de quatre termes principaux.

        **1. Séparation des nuages**

        Le terme principal cherche à maximiser la distance entre les nuages

        **2. Répartition des cardinalités**

        Pour chaque cardinalité admissible, un effectif différentiable est
        estimé. Puis une pénalité mesure l'écart entre ces effectifs et leur
        moyenne.

        **3. Régularisation du nombre de points**

        Le terme ``regw`` applique une barrière logarithmique aux nombres de
        points normalisés afin d'éviter les solutions trop proches des bornes.

        **4. Régularisation des poids**

        La quantité repul_z encourage les poids à s'éloigner de zéro, 
        ce qui favorise des masques de sélection plus tranchés.

        Les coefficients ``pds1``, ``pds2`` et ``pds3`` sont recalibrés lors
        du premier appel puis tous les 50 appels tant que ``it < 2000``.

    Warning:
        Si ``regw`` devient indéfini (NaN), une pénalité de secours 
        est utilisée afin de conserver une fonction de coût exploitable.
    """
    counter_w_opt +=1
    regw = torch.mean(torch.stack([outils.bornew((P-Nmin)/(Nmax-Nmin)) for P in nb_points]))
    # Régularisation taille
    if torch.isnan(regw):
        penalty_size = 0
        for nb in nb_points:
            penalty_size += torch.relu(Nmin - nb)  
    effect = torch.zeros(num_sizes, device=cloud_list[0].device)
    for i in range(num_sizes):
        effect[i] = outils.soft_count_near(nb_points,nb_pnt_possible[i],temp) 
        
    dif_att_rll = torch.stack([( (effect.sum()/num_sizes) - efi  )**2 for efi in effect])
    pena = torch.mean(dif_att_rll)
    
    repul_z = 0
    for w in weight_list:
        repul_z += outils.w_penal(w)
    repul_z = repul_z/nb_samples
    
    if counter_w_opt == 1 or counter_w_opt% 50 == 0 and it<2000:
         with torch.no_grad():
             pds1 = torch.min((beta/alpha) * torch.abs(loss_dist)/pena, torch.tensor(.1)) * pds0
             pds2 = torch.min((gamma/alpha) * torch.abs(loss_dist)/regw,  torch.tensor(mu)) * pds0
             pds3 = torch.min((delta/alpha) * torch.abs(loss_dist)/repul_z, torch.tensor(1e-5)) * pds0
    
    if torch.isnan(pds2) and not torch.isnan(regw):
        with torch.no_grad():
            pds2 = torch.min((gamma/alpha) * torch.abs(loss_dist)/regw,  torch.tensor(mu)) * pds0
            print("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm")

    if torch.isnan(regw):
        return (- pds0 * loss_dist + pds0 * 1000 * penalty_size + pds1 * pena + pds3 * repul_z ), counter_w_opt, pds1, pds2, pds3
    else:   
        return (- pds0 * loss_dist + pds1 * pena +  max((20/(it+1)), 1e-2)* pds2 * regw + pds3 * repul_z), counter_w_opt, pds1, pds2, pds3


#Fonction principale d'optimisation multi-nuages par gradient alterné sur les positions et les poids de sélection.
def optim_boucl(cloud_list, weight_list, Nmin, Nmax, nb_samples, echdist = [], born_inf = torch.tensor([0,0]), 
                born_sup = torch.tensor([1,1]), d = 2, mu0 = 7e-4, born_disper_inf = 0, temp = 2.3, 
                plot_hist = False, inert_pena_ch = True, jln_mth = False, tol = 1e-6, repul_param = 0, Wasserstein = True):
    """
    Exécute la boucle principale d'optimisation des nuages de points.

    Cette fonction alterne entre l'optimisation des positions des points et
    celle des poids de sélection afin de construire un ensemble de nuages
    aussi différents que possible tout en respectant plusieurs contraintes
    géométriques et de cardinalité.

    L'optimisation repose sur deux fonctions de coût distinctes :

    - une perte dédiée aux positions des points ;
    - une perte dédiée aux poids de sélection.

    Le passage d'un problème à l'autre est piloté dynamiquement au cours des
    itérations par :func:`variable_changeante`.

    La séparation entre les nuages peut être évaluée soit à l'aide d'une
    distance de transport optimal régularisée (Sinkhorn), soit à l'aide d'une
    distance de type Maximum Mean Discrepancy (MMD).

    Args:
        cloud_list (list[torch.Tensor]): Liste des nuages de points à
            optimiser. Chaque élément est un tenseur de forme
            ``(Nmax, d)``.
        weight_list (list[torch.Tensor]): Liste des vecteurs de poids
            associés aux nuages.
        Nmin (int): Nombre minimal de points autorisé.
        Nmax (int): Nombre maximal de points autorisé.
        nb_samples (int): Nombre de nuages optimisés simultanément.
        echdist (torch.Tensor, optional): Distribution de référence utilisée
            lorsque ``jln_mth=True``.
        born_inf (torch.Tensor, optional): Borne inférieure du domaine.
        born_sup (torch.Tensor, optional): Borne supérieure du domaine.
        d (int, optional): Dimension de l'espace. Défaut : ``2``.
        mu0 (float, optional): Coefficient initial de régularisation.
            Défaut : ``7e-4``.
        born_disper_inf (float, optional): Borne inférieure utilisée lors
            de l'évaluation de la dispersion interne. Cette quantité est
            également réutilisée lors du calcul de certaines pénalisations
            de répulsion. Défaut : ``0``.
        temp (float, optional): Température utilisée pour le comptage
            différentiable des cardinalités. Une valeur élevée augmente la
            précision du comptage mais peut manquer certaines contributions ;
            une valeur plus faible rend le comptage plus lisse mais peut
            attribuer un même nuage à plusieurs cardinalités voisines.
            Défaut : ``2.3``.
        plot_hist (bool, optional): Affiche différents graphiques de suivi en
            fin d'optimisation. Défaut : ``False``.
        inert_pena_ch (bool, optional): Active la pénalité visant à imposer
            une répartition homogène des distances minimales intra-nuages.
            Défaut : ``True``.
        jln_mth (bool, optional): Active la méthode expérimentale fondée sur
            une distribution cible de distances. Défaut : ``False``.
        tol (float, optional): Tolérance utilisée dans les critères d'arrêt.
            Défaut : ``1e-6``.
        repul_param (float, optional): Paramètre de la pénalité de répulsion
            entre les points d'un même nuage. Défaut : ``0``.
        Wasserstein (bool, optional): Si ``True``, la séparation entre les
            nuages est évaluée à l'aide de la distance de Sinkhorn.
            Si ``False``, une distance de type MMD est utilisée.
            Défaut : ``True``.

    Returns:
        tuple:
            - **cloud_list** (*list[torch.Tensor]*) : nuages optimisés ;
            - **weight_list** (*list[torch.Tensor]*) : poids optimisés.

    Raises:
        ValueError: Si ``jln_mth=True`` et que ``echdist`` n'est pas fourni.
    """
    
    if jln_mth and len(echdist) == 0:
        raise ValueError("echdist doit etre ranseigner et non vide pour cette methode(jln_mth)")
    #initialisation de toutes les variable necessaire au bon fonctionnement de la boucle d'optimisation 
    #ainsi que celle permettant de recuperer les historique de certaine de nos variable
    it = 0
    peri_p = 0
    peri_w = 0
    alpha = 0.70
    beta = 0.20
    gamma = 0.03
    delta =0.07
    pds0 = min(np.log(nb_samples+1e-8), 10)
    prev_loss = 1000
    prev_ploss = 1000
    prev_wloss = 1000
    prev_prev_wloss = 10000
    prev_prev_ploss = 10000
    prev_loss_dist = 10000
    hst_lossw = []
    amplificateur = 1
    cntlds = 0
    hist_loss_cl = []
    ct = 0
    K = Nmax - Nmin + 1
    cnt = 0
    pds1 = pds2 = pds3 = 1
    nbp_possible = torch.linspace(Nmin, Nmax, Nmax - Nmin +1, device=device)
    
    sinkhorn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur= 0.15) # Wassersteine pour calculer les distance inter nuage

    ofr_comp = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur= 0.1,
        backend="tensorized") #Wasserstein pour comparer deux distribution pour jln_mth

    optimizer_P = torch.optim.AdamW(
        [P for P in cloud_list],
        lr=1e-3)
    
    if Nmin != Nmax:
        optimizer_w = torch.optim.AdamW(
            [w for w in weight_list],
            lr=1e-3)
    
    if len(born_inf)== d and len(born_sup) == d:
        born_disper = 1.2 * Nmax**(-1/d) *( torch.prod(born_sup-born_inf))**(1/d)
    else: 
        born_disper = 1.2 * Nmax**(-1/d)
    
    while it<30000 :
        turn_p, peri_p, peri_w, mu, amplificateur = variable_changeante(it, Nmin, Nmax, peri_p, peri_w, mu0, amplificateur)
        
        if turn_p:
            optimizer = optimizer_P
        else:
            optimizer = optimizer_w
        

        optimizer.zero_grad()

        P_masked, m_masked, nb_points = calc_point_poid_nb_inter(nb_samples, Nmax, d,
                                        cloud_list, weight_list, amplificateur, turn_p)
        
        loss_dist = calc_min_dist_inter_nuage(nb_samples, P_masked, m_masked, Nmin, Nmax, it, sinkhorn
                                              , ofr_comp, jln_mth, echdist, Wasserstein)
        
        min_intra_dist = dist_min_par_ech(nb_samples, cloud_list,P_masked, weight_list)
        
        if inert_pena_ch:
            inert_pena, born_disper_inf, born_disper = outils.cvm_uniform_loss(min_intra_dist,born_inf = born_disper_inf, born_sup =born_disper)
        else:
            inert_pena = 0
        
        if turn_p:
            if repul_param > 1e-5:
                pena_rep = torch.stack([outils.repulsion_penalty3(P, repul_param) for P in cloud_list]).mean()
            else:
                pena_rep = 0
            loss = calc_loss_p(born_inf,born_sup,d,cloud_list, pds0, loss_dist, pena_rep, nb_samples, inert_pena, mu0)
            if np.abs(prev_ploss - loss.item())<1e-5:
                peri_w = 10
                
            hist_loss_cl.append(loss.item())
        else:
            loss, ct, pds1, pds2, pds3 = calc_loss_w(Nmin, Nmax, nb_points, K, cloud_list, nbp_possible, temp,
                            weight_list, nb_samples, ct, it, alpha, beta, gamma, delta, loss_dist, mu
                            , pds0, pds1, pds2, pds3)

            if np.abs(prev_wloss - loss.item())<1e-5:
                peri_p = 10
         

        if it > 800 :
            if turn_p:
                if torch.abs(loss - prev_ploss) + np.abs(prev_wloss - prev_prev_wloss)< tol:
                    break
            else :
                if torch.abs(loss - prev_wloss) + np.abs(prev_ploss - prev_prev_ploss)< tol:
                    break
                
            if torch.abs(loss_dist - prev_loss_dist)< tol/2 and it > 1500:
                cntlds +=1
                if cntlds >7:
                    break
            else:
                cntlds = 0
        
        if peri_w >0 and torch.abs(loss - prev_wloss)<tol*10:
            peri_w = 0
            
        if peri_p >0 and torch.abs(loss - prev_ploss)<tol*10:
            peri_p = 0
        
        loss.backward()

        if turn_p: 
            torch.nn.utils.clip_grad_norm_(
                [p for p in (cloud_list)],
                torch.min(born_sup))
        else:
            torch.nn.utils.clip_grad_norm_(
                [p for p in (weight_list)],
                1.)
            
        optimizer.step()

        with torch.no_grad():
            for P in cloud_list:
                P.clamp_(born_inf + 1e-6, born_sup - 1e-6)
            
          
        if cnt>49:
            print(f"it {it} | loss {prev_ploss + prev_wloss:.6f} | loss w {prev_wloss:.6f} | loss cl {prev_ploss:.6f}")
            print(cloud_list[0].device)
            print(torch.cuda.memory_allocated() / 1e6, "MB")
            cnt = 0
        cnt += 1
        if turn_p:  
            prev_prev_ploss = prev_ploss
            prev_loss = loss.item() + prev_wloss + 1 * peri_p
            prev_ploss = loss.item()
        else:
            prev_prev_wloss = prev_wloss
            prev_loss += prev_wloss 
            prev_wloss = loss.item()
            
            hst_lossw.append(prev_wloss)
        prev_loss_dist = loss_dist
        it +=1
    
    if not jln_mth:
        sorted_intra_min, indice = torch.sort(min_intra_dist)
        indice = indice.cpu()
    
        cloud_list = [cloud_list[i] for i in indice]
        weight_list = [weight_list[i] for i in indice]
        
    if plot_hist:
        plt.hist(min_intra_dist.detach().cpu().numpy())
        plt.show()
        
        plt.plot(hst_lossw[500:])
        plt.show()
        
        plt.plot(hist_loss_cl[500:])
        plt.show()
    return cloud_list, weight_list