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
Avec :
- it l’itération à laquelle se trouve l’algorithme
- loss la valeur de la fonction loss à cette itération
- loss w la valeur de la loss porter sur l'optimisation des poids à à l’itération it
- loss cl la valeur de la loss porter sur l'optimisation de la position des point à l’itération it
- cuda:0 indique que le GPU est utilisé
- 17.258496 MB correspond à la mémoire GPU consommée

Le scripte étant pensé pour tourner sur gpu, nous vous invitons si possible à faire en sorte qu'il tourne sur gpu.


## Execusion
Pour générer votre propre échantillon de nuage de points à partir de ces codes, plusieurs option s'offre à vous. Le plus simple étant de faire :
```python
import exmple_appel

P_list_f = exmple_appel.creation_dun_ech(Nmin, Nmax, nech, a, b, d)
```
Avec :
- Nmin le nombre minimum de point dans les nuages générés
- Nmax le nombre maximum de point dans les nuages générés
- nech le nombre de nuage de points à générer
- a la borne inférieur de l’hyper-pavé dans lequel les nuages de points vivent
- b la borne supérieur de l’hyper-pavé dans lequel les nuages de points vivent
- d la dimension dans laquelle les nuage de points vivent


Il convient toutefois de se souvenir que le temps de calcule necessaire à l'obtention de l'echantillon augmente de manière quadratique avec l'augmentation du nombre de nuage dans l'échantillon ainsi qu'avec l'augmentation du nombre de points maximum dans les nuage. Ainsi, si l'on veux générer de très grand echantillon de nuage de points avec ces scripte, cela peux prendre beaucoup de temps. A titre indicatif, lorsque le code tourne sur gpu, le temps necessaire pour générer 100 nuage de points contenant entre 28 et 35 points en dimention 2 est d'environs 4 minutes.


## Utilisation et interpretation des sorties
L'echantillon de nuage de points que l'on obtient en sortie de ce script est sous la forme de liste de nuage de points ainsi l'objet P_list_f peux alors s'ecrire sous la forme :

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


De plus comme expiqué précedement si l'on laisse l'argument save = True alors les sortie de l'algorithme sont enregistré dans le dossier voulu, par defaut le dossier resultat_optim. Si le dossier est déjà existant alors l'algorithme vous demandera si il peut le supprimer ou non. Si vous lui donné l'autorisation alors le dossier sera supprimer et tout ce qui est dans le dossier sera perdu. Si l'on ne lui auorise pas alors il ne sauvegarde simplement aucune image mais les affichera tout de meme. 

Les image sauvegarder represente :
- la repartition des cardinaux des nuage
- la projection des nuage de points dans chaque paire de dimention de l'espace visé
- une matrice rassemblant chacune des projection dans les different plan precedement enregistrer
- Si les nuage sont en dimension 2 ou 3, chacun des nuage de points par batch de 9 nuages

## Les méthodes d'echantillonnage disponible
Plusieurs méthode d'échantillonnage sont disponnible avec ce scripte. Pour choisir la méthode utilisé par la fonction creation_dun_ech il faut joué sur les paramètre suivant :
- jln_mth initialement à False, permet d'échantillonné selon la méthode jln
- inert_pena_ch initialement à True, permet d'échantillonner selon la méthode de répulsion adaptative
- repul_param initialement à 0, permet d'imposé un écartement minimal entre les points dans chaque nuage de l'échantillon

Si jln_mth et inert_pena_ch sont tout deux à False et repul_param est à 0, alors l'échantillon obtenu en sortie de l'algorithme est celui obtenue avec un maximin sans pénalisation.

Pour chacune de ces méthode, il est possible de choisir une initialisation par LHS ou un initialisation par un simple tirage uniforme. Cependant, nous préconisons d'utiliser le lhs uniquement pour la méthode jln ou l'on observe une convergence plus rapide en moyenne, la ou l'on observe plutot le contraire avec les méthode basé sur le maximin. Mais quelque soit l'initialisation les méthode converge et resorte des résultat satisfaisant.


Il est également possible d'optenir un échantillon avec la méthode par LHS qui est utiliser pour l'initialisation de certaine de nos méthode. Pour ceci il suffit d'utiliser le code suivant :

```python
import initialisation

X0_lhs, _ = initialisation.initialisation(Nmin, Nmax, nb_sample, born_inf, born_sup, d
                 all_opt = False, aff = False, for_torch = False, lhs = True)
```
Avec les 6 premières variables ayant la meme fonction que dans l'exemple d'execusion precedent et les variable suivante ayant pour but de renvoyer le nuage de points dans le meme forma que la sortie de l'autre code.


## Exemple d'appelle









Pour tout problèmes ou question suplementaire je vous invite à me contacté à l'adresse mail loick.diener@gmail.com, ou à lire les doc-string prensente dans les scripte, j'essayerais de vous répondre au plus vite.










