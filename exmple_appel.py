import numpy as np
import torch
import affichage
import initialisation
import optim
import time
import warnings
import os
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_born(a, d):
    if len(a)!= d:
        warnings.warn("Attention : dimention de a differente de d, clone de a affin de comble la difference")
        born = torch.zeros(d)
        indic = 0
        for i in range(d):
            if indic<len(a):
                born[i] = a[indic]
            else:
                indic = 0
                born[i] = a[indic]
            indic +=1
    else:
        born = torch.tensor(a)
    return born

def mise_enforme_borne(a, b):
    """
    Normalise un domaine borné afin que sa plus petite dimension soit égale à 1.

    Cette fonction effectue un changement d'échelle du domaine défini par les
    bornes ``a`` et ``b``. Les coordonnées sont ramenées à un domaine
    équivalent dont la borne inférieure est nulle et dont la plus petite
    longueur caractéristique vaut 1.

    Cette transformation est notamment utile pour rendre les paramètres
    d'optimisation indépendants de l'échelle physique du domaine étudié.

    Args:
        a (torch.Tensor): Borne inférieure du domaine, de forme ``(d,)``.
        b (torch.Tensor): Borne supérieure du domaine, de forme ``(d,)``.

    Returns:
        tuple:
            - **new_a** (*torch.Tensor*) : borne inférieure normalisée,
              égale à un vecteur nul ;
            - **new_b** (*torch.Tensor*) : borne supérieure normalisée ;
            - **retour_param** (*torch.Tensor*) : facteur d'échelle utilisé
              pour la normalisation.

    Raises:
        ValueError: Si au moins une composante de ``b - a`` est négative.
    """
    temp = b - a
    retour_param = torch.min(temp)
    if retour_param<=0:
        raise ValueError("b doit etre la borne sup et ne pas etre egal a a")
    return torch.zeros(len(a)), temp/retour_param, retour_param
    
def remise_a_niveau(a, b, a_reel, retour_param, nech, P_list):
    """
    Ramène un ensemble de nuages de points dans leur domaine d'origine après
    une phase de normalisation.

    Cette fonction applique l'opération inverse de celle réalisée par
    :func:`mise_enforme_borne`. Les nuages de points normalisés sont
    redimensionnés puis translatés afin de retrouver les coordonnées du
    domaine initial.

    Args:
        a (torch.Tensor): Borne inférieure du domaine normalisé.
            Ce paramètre est remplacé par ``a_reel`` dans la fonction.
        b (torch.Tensor): Borne supérieure du domaine normalisé.
        a_reel (torch.Tensor): Borne inférieure du domaine original.
        retour_param (float ou torch.Tensor): Facteur d'échelle retourné par
            :func:`mise_enforme_borne`.
        nech (int): Nombre de nuages contenus dans ``P_list``.
        P_list (list[torch.Tensor]): Liste des nuages de points normalisés.

    Returns:
        tuple:
            - **a** (*torch.Tensor*) : borne inférieure du domaine original ;
            - **b** (*torch.Tensor*) : borne supérieure du domaine original ;
            - **P_list** (*list[torch.Tensor]*) : nuages remis dans leur
              système de coordonnées initial.
    """
    a = a_reel
    for i in range(nech):
        P_list[i] = P_list[i]*retour_param + a
    b = b*retour_param
    b += a
    return a,b, P_list
    

#comportement pouvant etre etrange si une des dimention est beaucoup plus petites que les autre de l'ordre de 1 pour 100 voir 1 pour 1000
def creation_dun_ech(Nmin, Nmax, nech, a = torch.tensor([0,0]), b = torch.tensor([1,1]), d= 2,nrechlhs = 60000, Wasserstein = True, jln_mth = False,
                     mu0 = 7e-4, temp = 3.5, plot_hist = False, inert_pena_ch = True, tol = 1e-6, repul_param = 0,
                     all_opt = True, aff = True, for_torch = True, seed = None, veux_coin = False, lhs = False, aff_repart = True,
                     dossier = "resultat_optim", save = False,  export_all = False, aff_fin_nage = True, aff_sup_nuage = True):
    """
    Génère un ensemble optimisé de nuages de points.

    Cette fonction constitue le point d'entrée principal du processus de
    génération. Elle réalise successivement :

    1. la normalisation du domaine d'étude ;
    2. la génération des configurations initiales ;
    3. l'initialisation des poids de sélection ;
    4. l'optimisation des positions et des poids ;
    5. la remise à l'échelle dans le domaine original ;
    6. le traitement, l'affichage et l'export éventuel des résultats.

    Args:
        Nmin (int): Nombre minimal de points autorisé dans un nuage.
        Nmax (int): Nombre maximal de points autorisé dans un nuage.
        nech (int): Nombre de nuages à générer.
        jln_mth (bool, optional): Active la méthode expérimentale fondée
            sur une distribution cible de distances. Défaut : ``False``.
        a (torch.Tensor, optional): Borne inférieure du domaine de travail.
        b (torch.Tensor, optional): Borne supérieure du domaine de travail.
        d (int, optional): Dimension de l'espace. Défaut : ``2``.
        nrechlhs (int, optional): Nombre de recherches effectuées lors de la
            construction du plan Latin Hypercube de type maximin.
            Défaut : ``60000``.
        Wasserstein (bool, optional): Si ``True``, la séparation entre les
            nuages est évaluée à l'aide d'une distance de transport optimal
            (Sinkhorn). Sinon, une distance de type MMD est utilisée.
            Défaut : ``True``.
        mu0 (float, optional): Coefficient initial de régularisation.
            Défaut : ``7e-4``.
        temp (float, optional): Température utilisée dans le comptage
            différentiable des cardinalités. Défaut : ``2.3``.
        plot_hist (bool, optional): Affiche des graphiques de diagnostic
            après optimisation. Défaut : ``False``.
        inert_pena_ch (bool, optional): Active la pénalité de répartition
            des distances minimales intra-nuages. Défaut : ``True``.
        tol (float, optional): Tolérance utilisée dans les critères
            d'arrêt. Défaut : ``1e-6``.
        repul_param (float, optional): Paramètre de la pénalité de
            répulsion interne aux nuages. Défaut : ``0``.
        all_opt (bool, optional): Si ``True``, tous les nuages sont
            initialisés avec ``Nmax`` points. Défaut : ``True``.
        aff (bool, optional): Affiche les nuages initiaux. Défaut :
            ``True``.
        for_torch (bool, optional): Active le calcul des gradients sur les
            points générés. Défaut : ``True``.
        seed (int, optional): Graine de reproductibilité.
        veux_coin (bool, optional): Ajoute des configurations initiales
            situées dans les coins du domaine. Défaut : ``False``.
        lhs (bool, optional): Utilise une initialisation basée sur un plan
            Latin Hypercube. Défaut : ``False``.
        aff_repart (bool, optional): Affiche la répartition finale des
            cardinalités. Défaut : ``True``.
        dossier (str, optional): Répertoire de sauvegarde des résultats.
            Défaut : ``"resultat_optim"``.
        save (bool, optional): Sauvegarde les figures produites.
            Défaut : ``False``.
        export_all (bool, optional): Exporte les nuages optimisés dans un
            fichier texte. Défaut : ``False``.
        aff_fin_nage (bool, optional): Affiche les nuages optimisés
            individuellement. Défaut : ``True``.
        aff_sup_nuage (bool, optional): Affiche les projections globales
            des nuages optimisés. Défaut : ``True``.

    Returns:
        list[torch.Tensor]: Liste des nuages de points optimisés dans le
        domaine d'origine.

    Notes:
        Le domaine défini par ``[a, b]`` est d'abord normalisé à l'aide de
        :func:`mise_enforme_borne` afin de rendre les hyperparamètres de
        l'algorithme moins sensibles à l'échelle du problème.

        Lorsque ``Nmin != Nmax``, un vecteur de poids est associé à chaque
        nuage. Ces poids sont optimisés conjointement aux positions des
        points afin de déterminer automatiquement le nombre effectif de
        points conservés.

        L'optimisation est réalisée par :func:`optim.optim_boucl`
        
        Les résultats sont ensuite réinjectés dans le domaine d'origine via
        :func:`remise_a_niveau`.
    """
    
    if save and os.path.exists(dossier):
    
        while True:
            reponse = input(
                f"Le dossier '{dossier}' existe déjà. Remplacer ? (o/n) : "
            ).strip().lower()
    
            if reponse in ("o", "oui"):
                shutil.rmtree(dossier)  # suppression du dossier et de son contenu
                break
    
            elif reponse in ("n", "non"):
                save = False
                break
    
            else:
                print("Réponse invalide. Entrez 'o' ou 'n'.")
    
    if save:
        os.makedirs(dossier)
        
    
    a = init_born(a, d)
    b = init_born(b, d)
    a_reel = a.clone()
    a, b, retour_param = mise_enforme_borne(a, b)
    P_list, echdist = initialisation.initialisation(Nmin, Nmax, nech,jln_mth, a, b, d, nrechlhs, all_opt, aff, for_torch, seed, veux_coin, lhs)
    if Nmin != Nmax:
        w_list = [(torch.rand(Nmax, requires_grad=True, device=device)) for _ in range(nech)]

        for w in w_list:
            ind = np.random.choice(np.arange(0, Nmax), size= np.random.randint(0, Nmax - Nmin + 1), replace=False)
        
            with torch.no_grad():
                w[ind] = -w[ind]
    else: 
        w_list = []
    P_list, w_list = optim.optim_boucl(P_list, w_list, Nmin, Nmax, nech, echdist, a, b, 
                    d, mu0, temp, False, plot_hist, inert_pena_ch, 
                    jln_mth, tol, repul_param, Wasserstein)
    
    a, b, P_list = remise_a_niveau(a, b, a_reel, retour_param, nech, P_list)
    P_final = affichage.traitement_et_aff(cloud_list = P_list, weight_list = w_list, Nmin = Nmin, 
                      Nmax = Nmax, d = d, born_inf = a, born_sup = b, export_all = export_all,
                      aff_fin_nage = aff_fin_nage, aff_sup_nuage = aff_sup_nuage,
                      save = save, dossier = dossier, aff_repart = aff_repart)
    
    for i in range(nech):
        P_final[i] = P_final[i].cpu().numpy()
    
    return P_final #si tu veux en faire qqc directement ici
    
    

if __name__ == '__main__':
    start = time.perf_counter()
    torch.set_default_dtype(torch.float32) #a set toujours avant d'appeler la fonction attention beaucoup plus rappide en float32 qu'en float64
    P_list_f = creation_dun_ech(Nmin = 18, Nmax = 25, nech = 100, a = torch.tensor([200.365,-4250.2154]), b = torch.tensor([208.365,-4242.2154]), d = 2,
                     plot_hist = True, inert_pena_ch = True, lhs = False)

    end = time.perf_counter()
    print(f"Temps d'exécution : {end - start:.6f} secondes")
    
    