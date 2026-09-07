# Sampling_point_cloud

Sampling_point_cloud est un ensemble de méthode d'échantillonnage sur l'ensemble des nuage de points.

## Installation
Avant de pouvoir lancer les différent script python, il est nécessaire de set-up un environnement conda en utilisant les commandes suivante :

```bash
conda create -n your_env_name python=3.12 numpy=2.5.2 matplotlib=3.11.1 pytorch=2.6.0 bioconda::geomloss=0.3.1
conda activate your_env_name
```
Avec your_env_name le nom de l'environnement que vous voulez créer. 

La commande ci-dessus installe python 3.12 dans l'environnement mais si vous avez une quelconque autre préférence ces scripte fonctionne avec les version de python de 3.10 à 3.14.3. Pour ce qui est d'autre version le teste n'a pas été effectué.

## Validation

Pour confirmer que l'ensemble des librairies soit correctement installé nous vous conseillons d'effectuer la commande :

```bash
python ./exmple_appel.py
```
Les nuage de points générer par cette appel seront alors présent dans le dossier resultat_optim nouvellement créé

Au cour de l'execution vous verrez apparaitre dan sle terminal :
```bash
it 50 | loss 911194.534353 | loss w 911193.312500 | loss cl 1.221853
cuda:0
17.258496 MB
```
Où it correspond à l’itération à laquelle l’algorithme se trouve, loss correspond à la valeur de la fonction loss à cette itération, loss w correspond à la valeur de la loss porter sur l'optimisation des poids à à l’itération it, loss cl correspond à la valeur de la loss porter sur l'optimisation de la position des point à l’itération it. Pour ce qui est des inscription cuda:0 et 17.258496 MB elle apparaisse si l'installation à été fait correctement et si votre machine contient un gpu. L'optimisation s'effectuant beaucoup plus rapidement sur gpu que sur cpu. Ainsi, 17.258496 MB correspond à la mémoire utilisé sur le gpu par l'algorithme.


## Execusion
Pour générer votre propre échantillon de nuage de points à partir de ces codes, plusieurs option s'offre à vous. Le plus simple étant de faire :
```python
import exmple_appel

P_list_f = exmple_appel.creation_dun_ech(Nmin, Nmax, nech, a, b, d)
```

Avec, à modifier selon votre objectif, Nmin et Nmax respectivement le nombre de points maximum et minimum de point dans les nuages générés, nech le nombre de nuage de points à générer, a la borne inférieur de l’hyper-pavé dans lequel les nuages de points vivent, b la borne supérieur de l’hyper-pavé dans lequel les nuages de points vivent et d la dimension dans laquelle les nuage de points vivent. Il convient toutefois de se souvenir que la complexité algorithmique du scripte actuelle est en complexité quadratique et qu'ainsi le temps de calcule peux devenir très long lorsque l'on essaye de générer un grand nombre de nuage de ponts. A titre indicatif, lorsque le code tourne sur gpu, le temps necessaire pour générer 100 nuage de points contenant entre 28 et 35 points en dimention 2 est d'environs 4 minutes.

Par défaut la méthode échantillonnage utiliser est la méthode par répulsion adaptative, si une autre méthode d'échantillonnage est souhaité il faut jouer sur les variable jln_mth initialement à False permet de choisir ou non la méthode jln, inert_pena_ch initialement à True permet de choisir ou non la méthode avec répulsion adaptative, repul_param initialement à 0 et etant un réèl permet de définir un écartement minimal entre deux points d'un nuage. Ainsi si chacune de ces variable est à False ou à 0 alors la méthode maximin sans penalisation est alors lancé. 

Enfin si vous voulez simplement obtenir les nuage de points obtenue avec la méthode par LHS décrite dans le rapport, il vous suffit d'executé le code :

```python
import initialisation

X0_lhs, _ = initialisation.initialisation(Nmin, Nmax, nb_sample, born_inf, born_sup, d
                 all_opt = False, aff = False, for_torch = False, lhs = True)
```
Avec les 6 premières variables ayant la meme fonction que dans l'exemple d'execusion precedent et les variable suivante etant simplement pour ne pas obtenir le formalisme necessaire au lancement de l'optimisation sur les nuage de points, ce qui compliquerait inutiliement l'affichage et la comprehension des nuage en sortie.

Le choix de l'initialisation est également possible pour les différente méthode d'echantillonnage contenant une optimisation. Nous préconisons d'utiliser le lhs uniquement pour la méthode jln ou l'on observe une convergence plus rapide en moyenne, la ou l'on observe plutot le contraire avec les méthode basé sur le maximin. Mais quelque soit l'initialisation les méthode converge et resorte des résultat satisfaisant.


## Utilisation et interpretation des sorties
Chacune des méthode retourne alors l'echantillon de nuage de points sous forme de liste de nuage de points ainsi l'objet P_list_f ou X0_lhs peux alors s'ecrire sous la forme :

```bash
[C_1, C_2, ..., C_N]
```

Avec C_j le j-ème nuage de points de l'échantillon représenté par un numpy.ndarray qui s'écrit sous la forme :

```bash
[P_1, P_2, ..., P_n]
```

Et ou P_i représente le i-ème points du nuage de points représenté par un numpy.ndarray qui s'écrit sous la forme :
```bash
[c_1, c_2, ..., c_d]
```
Avec c_k la k-ième coordonées du point.


De plus comme expiqué précedement si l'on laisse l'argument save = True alors les sortie de l'algorithme sont enregistré dans le dossier voulu, par defaut le dossier resultat_optim. Si le dossier est déjà existant alors l'algorithme vous deemandera si il peut le supprimer ou non. Si vous lui donné l'autorisation alors le dossier sera supprimer et tout ce qui est dans le dossier sera perdu. Si l'on ne lui auorise pas alors il ne sauvegarde simplement aucune image mais les affichera tout de meme. Les image sauvegarder represente la repartition des cardinaux des nuage en fin de boucle, la projection des nuage de points dans chaque paire de dimention de l'espace visé (la première dimention etant la dimention 0), une matrice rassemblant chacune des projection dans les different plan precedement enregistrer avec en plus la repartition des points sur chaque axe ainsi que, si les nuage sont en dimention 2 ou 3, chacun des nuage de points afficher par batch de 9 nuages.

Pour tout problèmes ou question suplementaire je vous invite à me contacté à l'adresse mail loick.diener@gmail.com, ou à lire les doc-string prensente dans les scripte, j'essayerais de vous répondre au plus vite.










