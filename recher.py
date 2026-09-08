"""
Ce bout de code permet de récupérer, sous un forma identique au forma en sortie des scripte,
l'échantillon de nuage de points qui est sauvegarder sous le nom array_X0s.txt lorsque l'on 
decide de l'exporter. Tel quel le code ne fait evidement rien de plus.
"""

import os
import re
import numpy as np


def load_pnt_cl(dossier = "resultat_optim", fichier_in = "array_X0s.txt"):
    """
    Recharge une liste de tableaux NumPy depuis un fichier texte contenant
    une succession de matrices enregistrées sous forme de chaînes de
    caractères.
    Args:
        dossier (str): Répertoire contenant le fichier à lire.
            Par défaut : ``"resultat_optim"``.

        fichier_in (str): Nom du fichier texte contenant les matrices.
            Par défaut : ``"array_X0s.txt"``.
            
    Returns:
        list[np.ndarray]:
            Liste des tableaux NumPy reconstruits à partir du contenu
            du fichier. Chaque élément de la liste correspond à une
            matrice enregistrée dans le fichier et les dimensions peuvent
            varier d'un élément à l'autre. Typiquement un nuage de points.

    Notes:
        Cette fonction est conçue pour relire les fichiers contenant les 
        échantillon de nuage de points généré par les autre scripte de ce 
        dépot git.
    """
    fichier = os.path.join(dossier, fichier_in)
    X0_list = []
    with open(fichier, "r") as f:
        content = f.read()
      
    blocks = re.findall(r"\[\[.*?\]\]", content, flags=re.DOTALL)
    
    for block in blocks:
        block = block.replace("[[", "").replace("]]", "")
        rows = []
        for line in block.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = line.replace("[", "").replace("]", "")
            rows.append(np.fromstring(line, sep=" "))
        X0_list.append(np.vstack(rows))
    return X0_list


