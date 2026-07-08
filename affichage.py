import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy.random as rng
import outils
from geomloss import SamplesLoss
import random
import time
import os
import initialisation



def aff_init(ech, a,b,batch_size, ncols, nrows, titre, save = False, dossier = "resultat_optim"):
    # affichage de nos tirages par batch de 5 * 5

    nech = len(ech)  # sécurité
    truc = 0
    for batch_start in range(0, nech, batch_size):
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15))
        axes = axes.flatten()
        for k in range(batch_size):
            j = batch_start + k
            ax = axes[k]
            if j >= nech:
                ax.axis("off")
                continue
            samples = ech[j]        # Tensor (n, d)
            x_coords = samples[:, 0].detach().cpu().numpy()
            y_coords = samples[:, 1].detach().cpu().numpy()
            ax.scatter(x_coords, y_coords, s=10)
            axes[k].set_xlim(a[0].cpu() - (b[0].cpu() - a[0].cpu()) *0.05, b[0].cpu() + (b[0].cpu() - a[0].cpu()) * 0.05)
            axes[k].set_ylim(a[1].cpu() - (b[1].cpu() - a[1].cpu()) *0.05, b[1].cpu() + (b[1].cpu() - a[1].cpu()) * 0.05)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(dossier, f"{titre} {truc}.png"))
            plt.close()
        else:
            plt.show()
        truc +=1
        
        
        
        
def proj_sur_un_plan(d, P_optimals_pt, save, dossier):
    if d>1:
        for i in range(d-1):
            for j in range(i+1,d):
                for k, P in enumerate(P_optimals_pt):
                    P_np = P.detach().cpu().numpy()
                    plt.scatter(P_np[:, i], P_np[:, j], color='gray')
        
                plt.legend()
                plt.title(f"Nuages projeter dans les dimention {i} et {j}")
                if save:
                    plt.savefig(os.path.join(dossier, "All_nuage_opt.png"))
                    plt.close()
                else:
                    plt.show()
    else:
        plt.figure()
        for k, P in enumerate(P_optimals_pt):
            P_np = P.detach().cpu().numpy()
            plt.plot(P_np[:,0], np.zeros(len(P_np)),'o', color='gray')
        plt.yticks([])
        plt.title("Nuages de points")
        plt.legend()
        if save:
            plt.savefig(os.path.join(dossier, "All_nuage_opt.png"))
            plt.close()
        else:
            plt.show()       
        
        

def traitement_et_aff(P_list, w_list, nech, Nmin, Nmax, d, a = torch.tensor([0,0]), 
                      b = torch.tensor([1,1]), export_all = False,
                      aff_fin_nage = True, aff_sup_nuage = True,
                      save = False, dossier = "resultat_optim", aff_repart = True):
    K = Nmax - Nmin + 1
    P_optimals_pt = []
    nbpntpt = []
    with torch.no_grad():
        for P, w in zip(P_list, w_list):
            mask = w > 0
            P_optimals_pt.append(P[mask].to(torch.float64))
            nbpntpt.append(len(P[mask]))
            
    vnbppt = []
    for P in P_optimals_pt:
        vnbppt.append(len(P))
        
    if aff_sup_nuage or save:
        proj_sur_un_plan(d, P_optimals_pt, save, dossier)
        
        
    if aff_repart:
        plt.hist(vnbppt, bins = K)
        plt.title("Repartition fin de boucle")
        plt.xlim(Nmin,Nmax+1)
        if save:
            plt.savefig(os.path.join(dossier, "All_nuage_opt.png"))
            plt.close()
        else:
            plt.show()

    if aff_fin_nage or save:
        aff_init(P_optimals_pt , a,b, 9, 3, 3, "nuage_opti", save, dossier)
        
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
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                