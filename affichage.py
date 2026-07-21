import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from collections import Counter


    
    

def aff_init(ech, born_inf, born_sup, d, batch_size, nb_cols, nb_rows, 
             titre, save = False, dossier = "resultat_optim"):
    """
    Affiche ou sauvegarde des ensembles de nuages de points en dimension 2 ou 3.

    Args:
        ech (list[torch.Tensor]): Liste des nuages de points à visualiser.
            Chaque élément doit être un tenseur de forme
            ``(n_points, d)``.
        born_inf (torch.Tensor): Borne inférieure du domaine d'étude.
            Doit être de forme ``(d,)``.
        born_sup (torch.Tensor): Borne supérieure du domaine d'étude.
            Doit être de forme ``(d,)``.
        d (int): Dimension des données à afficher. Seules les dimensions
            ``2`` et ``3`` sont actuellement prises en charge.
        batch_size (int): Nombre maximum de nuages représentés sur une même
            figure.
        nb_cols (int): Nombre de colonnes de la grille d'affichage.
        nb_rows (int): Nombre de lignes de la grille d'affichage.
        titre (str): Préfixe utilisé pour le nom des figures sauvegardées.
        save (bool, optional): Si ``True``, les figures sont enregistrées dans
            ``dossier``. Sinon elles sont affichées à l'écran.
            Défaut : ``False``.
        dossier (str, optional): Répertoire de sauvegarde lorsque
            ``save=True``. Défaut : ``"resultat_optim"``.

    Returns:
        None

    Warning:
        Les dimensions autres que 2 ou 3 ne sont pas prises en charge. Dans
        ce cas, un message d'avertissement est affiché et aucun graphique
        n'est produit.
    """

    nb_sample = len(ech)  # sécurité
    indx_batch = 0
    if d == 2 :
        for batch_start in range(0, nb_sample, batch_size):
            fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(15, 15))
            axes = axes.flatten()
            for k in range(batch_size):
                j = batch_start + k
                ax = axes[k]
                if j >= nb_sample:
                    ax.axis("off")
                    continue
                samples = ech[j]        # Tensor (n, d)
                x_coords = samples[:, 0].detach().cpu().numpy()
                y_coords = samples[:, 1].detach().cpu().numpy()
                ax.scatter(x_coords, y_coords, s=10)
                ax.set_xlim(born_inf[0].cpu().numpy(), born_sup[0].cpu().numpy())
                ax.set_ylim(born_inf[1].cpu().numpy(), born_sup[1].cpu().numpy())
                ax.set_xlabel("x")
                ax.set_ylabel("y")
        
            plt.tight_layout()
            if save:
                plt.savefig(os.path.join(dossier, f"{titre} {indx_batch}.png"))
                plt.close()
            else:
                plt.show()
            indx_batch +=1
            
    elif d == 3:
        for batch_start in range(0, nb_sample, batch_size):
            fig = plt.figure(figsize=(15, 15))
            axes = []
            # création manuelle des subplots 3D
            for i in range(nb_rows * nb_cols):
                axes.append(fig.add_subplot(nb_rows, nb_cols, i+1, projection='3d'))
            for k in range(batch_size):
                j = batch_start + k
                ax = axes[k]
                if j >= nb_sample:
                    ax.axis("off")
                    continue
                samples = ech[j]
                x_coords = samples[:, 0].detach().cpu().numpy()
                y_coords = samples[:, 1].detach().cpu().numpy()
                z_coords = samples[:, 2].detach().cpu().numpy()
                ax.scatter(x_coords, y_coords, z_coords, s=10)
                ax.set_xlim(born_inf[0].cpu().numpy(), born_sup[0].cpu().numpy())
                ax.set_ylim(born_inf[1].cpu().numpy(), born_sup[1].cpu().numpy())
                ax.set_zlim(born_inf[2].cpu().numpy(), born_sup[2].cpu().numpy())
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
            plt.tight_layout()
            if save:
                plt.savefig(os.path.join(dossier, f"{titre} {indx_batch}.png"))
                plt.close()
            else:
                plt.show()
            indx_batch +=1
    else:
        print("dimention incompatible avec cette affichage attention")
        
        
def proj_sur_un_plan_matrice(d, cloud_list, save=False,
                             dossier="resultat_optim"):
    """
    Affiche ou sauvegarde une matrice de projections de plusieurs nuages de
    points.

    Cette fonction construit une matrice de graphiques de taille ``d × d``
    permettant de visualiser simultanément les distributions marginales et
    les projections bidimensionnelles d'un ensemble de nuages de points.

    Les éléments diagonaux de la matrice contiennent les histogrammes des
    coordonnées selon chaque dimension, tandis que les éléments hors
    diagonale représentent les projections des nuages sur les plans
    bidimensionnels correspondants.

    Args:
        d (int): Dimension des nuages de points.
        cloud_list (list[torch.Tensor]): Liste des nuages de points à
            visualiser. Chaque élément doit être un tenseur de forme
            ``(n_points, d)``.
        save (bool, optional): Si ``True``, la figure est sauvegardée dans le
            répertoire ``dossier``. Sinon, elle est affichée à l'écran.
            Défaut : ``False``.
        dossier (str, optional): Répertoire dans lequel sauvegarder la figure.
            Défaut : ``"resultat_optim"``.

    Returns:
        None
    """
    fig, axes = plt.subplots(d, d, figsize=(3*d, 3*d))
    if d == 1:
        axes = np.array([[axes]])
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                all_values = np.concatenate([
                    P.detach().cpu().numpy()[:, i]
                    for P in cloud_list])
                
                ax.hist(all_values, bins=20, color='gray')
                ax.set_title(f"dim {i}")
                continue
            for P in cloud_list:
                P_np = P.detach().cpu().numpy()
                ax.scatter(P_np[:, j], P_np[:, i], color='gray', s=10)
            if i == d - 1:
                ax.set_xlabel(f"dim {j}")
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(f"dim {i}")
            else:
                ax.set_yticklabels([])
                
    plt.suptitle("Matrix des projections", fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(dossier, "Matrice_projections.png"), bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()       
        
        
        
        
def proj_sur_un_plan(d, cloud_list, save=False,
                     dossier="resultat_optim"):
    """
    Affiche ou sauvegarde les projections bidimensionnelles de plusieurs
    nuages de points.

    Pour les espaces de dimension supérieure à 1, la fonction génère toutes
    les projections possibles sur les plans définis par les couples de
    dimensions ``(i, j)`` avec ``i < j``. Une matrice complète des
    projections est également produite via
    :func:`proj_sur_un_plan_matrice`.

    Dans le cas unidimensionnel, les points sont simplement représentés sur
    un axe horizontal.

    Args:
        d (int): Dimension des nuages de points.
        cloud_list (list[torch.Tensor]): Liste des nuages de points à
            visualiser. Chaque élément doit être un tenseur de forme
            ``(n_points, d)``.
        save (bool, optional): Si ``True``, les figures sont sauvegardées dans
            ``dossier``. Sinon, elles sont affichées à l'écran.
            Défaut : ``False``.
        dossier (str, optional): Répertoire utilisé pour la sauvegarde des
            figures. Défaut : ``"resultat_optim"``.

    Returns:
        None
    """
    if d>1:
        for i in range(d-1):
            for j in range(i+1,d):
                for k, P in enumerate(cloud_list):
                    P_np = P.detach().cpu().numpy()
                    plt.scatter(P_np[:, i], P_np[:, j], color='gray')
                plt.title(f"Nuages projeter dans les dimention {i} et {j}")
                if save:
                    plt.savefig(os.path.join(dossier, f"All_nuage_opt_dim_{i}_{j}.png"))
                    plt.close()
                else:
                    plt.show()
        proj_sur_un_plan_matrice(d, cloud_list, save, dossier)
    else:
        plt.figure()
        for k, P in enumerate(cloud_list):
            P_np = P.detach().cpu().numpy()
            plt.plot(P_np[:,0], np.zeros(len(P_np)),'o', color='gray')
        plt.yticks([])
        plt.title("Nuages de points")
        if save:
            plt.savefig(os.path.join(dossier, "All_nuage_opt.png"))
            plt.close()
        else:
            plt.show()      
    
        
    
def traitement_et_aff(cloud_list, weight_list, Nmin, Nmax, d, born_inf = torch.tensor([0,0]), 
                      born_sup = torch.tensor([1,1]), export_all = False,
                      aff_fin_nage = True, aff_sup_nuage = True,
                      save = False, dossier = "resultat_optim", aff_repart = True):
    """
    Traite les nuages optimisés, produit différentes visualisations et,
    éventuellement, exporte les résultats.

    Cette fonction applique les masques de sélection définis par ``weight_list``
    aux nuages de points contenus dans ``cloud_list`` afin de construire les
    nuages optimisés finaux. Elle peut ensuite afficher ou sauvegarder
    plusieurs représentations graphiques, notamment les projections des
    nuages, leur distribution en nombre de points et leur visualisation
    individuelle.

    Args:
        cloud_list (list[torch.Tensor]): Liste des nuages de points optimisés
            avant filtrage. Chaque élément est un tenseur de forme
            ``(Nmax, d)``.
        weight_list (list[torch.Tensor]): Liste des vecteurs de poids ou de masques
            associés aux nuages de ``cloud_list``.
        Nmin (int): Nombre minimal de points attendu dans un nuage final.
        Nmax (int): Nombre maximal de points dans un nuage.
        d (int): Dimension des nuages de points.
        born_inf (torch.Tensor, optional): Borne inférieure du domaine d'étude.
            Défaut : ``torch.tensor([0, 0])``.
        born_sup (torch.Tensor, optional): Borne supérieure du domaine d'étude.
            Défaut : ``torch.tensor([1, 1])``.
        export_all (bool, optional): Si ``True``, exporte les nuages traités
            dans un fichier texte. Défaut : ``False``.
        aff_fin_nage (bool, optional): Si ``True``, affiche les nuages
            optimisés individuellement. Défaut : ``True``.
        aff_sup_nuage (bool, optional): Si ``True``, affiche les projections
            globales des nuages optimisés. Défaut : ``True``.
        save (bool, optional): Si ``True``, sauvegarde les figures produites
            dans ``dossier``. Défaut : ``False``.
        dossier (str, optional): Répertoire de sauvegarde des résultats
            graphiques. Défaut : ``"resultat_optim"``.
        aff_repart (bool, optional): Si ``True``, affiche l'histogramme de la
            répartition du nombre de points des nuages optimisés.
            Défaut : ``True``.

    Returns:
        list[torch.Tensor]: Liste des nuages optimisés après application des
        masques de sélection.
    """
    K = Nmax - Nmin + 1
    P_optimals_pt = []
    nbpntpt = []
    with torch.no_grad():
        for P, w in zip(cloud_list, weight_list):
            mask = w > 0
            P_optimals_pt.append(P[mask].to(torch.get_default_dtype()))
            nbpntpt.append(len(P[mask]))
            
    vnbppt = []
    for P in P_optimals_pt:
        vnbppt.append(len(P))
        
    if aff_sup_nuage or save:
        proj_sur_un_plan(d, P_optimals_pt, save, dossier)
        
        
    if aff_repart:
        compte = Counter(vnbppt)

        x = sorted(compte.keys())
        y = [compte[k] for k in x]

        plt.bar(x, y)
        plt.xlabel("Nombre de points")
        plt.ylabel("Effectif")
        plt.title("Repartition fin de boucle")

        if save:
            plt.savefig(os.path.join(dossier, "repart_finb.png"))
            plt.close()
        else:
            plt.show()

    if aff_fin_nage or save:
        aff_init(P_optimals_pt , born_inf,born_sup, d, 9, 3, 3, "nuage_opti", save, dossier)
        
    if export_all:
        X0_list = []

        for P in P_optimals_pt:
            X0_temp = []
            for pnt in P:
                X0_temp.append(pnt.cpu().numpy().item())
            X0_list.append(X0_temp)
        
        dossier = "ensemble_X0_genetraiter"
        os.makedirs(dossier, exist_ok=True)

        with open(os.path.join(dossier, "array_X0s.txt"), "w") as f:
            for element in X0_list:
                f.write(str(element) + "\n")
    return P_optimals_pt
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                