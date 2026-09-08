# Sampling_point_cloud

Sampling_point_cloud est un ensemble de méthodes d'échantillonnage sur l'ensemble des nuage de points.

## Installation
Avant de pouvoir lancer les différents scripts Python, il est nécessaire de configurer un environnement conda en utilisant les commandes suivantes :

```bash
conda create -n your_env_name python=3.12 numpy=2.5.2 matplotlib=3.11.1 pytorch=2.6.0 bioconda::geomloss=0.3.1
conda activate your_env_name
```
Avec your_env_name le nom de l'environnement que vous voulez créer. 

La commande ci-dessus installe python 3.12 dans l'environnement, mais si vous avez une quelconque autre préférence, ces scripts fonctionnent avec les versions de python de 3.10 à 3.14.3. Pour ce qui est d'autres versions, le test n'a pas été effectué.

## Validation

Pour confirmer que l'ensemble des librairies soit correctement installé, nous vous conseillons d'effectuer la commande :

```bash
python ./exmple_appel.py
```
Les nuages de points générés par cet appel seront alors présents dans le dossier resultat_optim nouvellement créé.

Au cours de l'exécution, vous verrez apparaître dans le terminal :
```bash
it 50 | loss 911194.534353 | loss w 911193.312500 | loss cl 1.221853
cuda:0
17.258496 MB
```
Avec :
- it l’itération à laquelle se trouve l’algorithme
- loss la valeur de la fonction loss à cette itération
- loss w la valeur de la loss portant sur l'optimisation des poids à l’itération it
- loss cl la valeur de la loss portant sur l'optimisation de la position des points à l’itération it
- cuda:0 indique que le GPU est utilisé
- 17.258496 MB correspond à la mémoire GPU consommée

Le script étant pensé pour tourner sur gpu, nous vous invitons, si possible, à faire en sorte qu'il tourne sur gpu.


## Exécution
Pour générer votre propre échantillon de nuages de points à partir de ces codes, plusieurs options s'offrent à vous. La plus simple étant de faire :
```python
import exmple_appel

P_list_f = exmple_appel.creation_dun_ech(Nmin, Nmax, nech, born_inf, born_sup, d)
```
Avec :
- Nmin le nombre minimum de points dans les nuages générés
- Nmax le nombre maximum de points dans les nuages générés
- nech le nombre de nuages de points à générer
- born_inf la borne inférieure de l’hyper-pavé dans lequel les nuages de points vivent
- born_sup la borne supérieure de l’hyper-pavé dans lequel les nuages de points vivent
- d la dimension dans laquelle les nuages de points vivent


Il convient toutefois de se souvenir que le temps de calcul nécessaire à l'obtention de l'échantillon augmente de manière quadratique avec l'augmentation du nombre de nuages dans l'échantillon ainsi qu'avec l'augmentation du nombre de points maximum dans les nuages. Ainsi, si l'on veut générer de très grands échantillons de nuages de points avec ces scripts, cela peut prendre beaucoup de temps. A titre indicatif, lorsque le code tourne sur gpu, le temps nécessaire pour générer 100 nuages de points contenant entre 28 et 35 points en dimension 2 est d'environ 4 minutes.


## Utilisation et interprétation des sorties
L'échantillon de nuages de points que l'on obtient en sortie de ce script est sous la forme d'une liste de nuages de points ainsi l'objet P_list_f peut alors s'écrire sous la forme :

```bash
[C_1, C_2, ..., C_N]
```

Avec C_j le j-ième nuage de points de l'échantillon, représenté par un numpy.ndarray qui s'écrit sous la forme :

```bash
[P_1, P_2, ..., P_n]
```

Et où P_i représente le i-ième point du nuage de points, représenté par un numpy.ndarray qui s'écrit sous la forme :
```bash
[c_1, c_2, ..., c_d]
```
Avec c_k la k-ième coordonnée du point.


De plus, comme expliqué précédemment, si l'on laisse l'argument save = True alors les sorties de l'algorithme sont enregistrées dans le dossier voulu, par défaut le dossier resultat_optim. Si le dossier est déjà existant, alors l'algorithme vous demandera s'il peut le supprimer ou non. Si vous lui donnez l'autorisation alors le dossier sera supprimer et tout ce qui est dans le dossier sera perdu. Si l'on ne lui autorise pas alors il ne sauvegarde simplement aucune image, mais les affichera tout de même. 

Les images sauvegardées représentent :
- la répartition des cardinaux des nuages
- la projection des nuages de points dans chaque paire de dimensions de l'espace visé
- une matrice rassemblant chacune des projections dans les différents plans précédemment enregistrés
- Si les nuages sont en dimension 2 ou 3, chacun des nuage de points par batch de 9 nuages

## Les méthodes d'échantillonnage disponibles
Plusieurs méthodes d'échantillonnage sont disponibles avec ce scripte. Pour choisir la méthode utilisée par la fonction creation_dun_ech il faut jouer sur les paramètres suivants :
- jln_mth initialement à False, permet d'échantillonner selon la méthode jln
- inert_pena_ch initialement à True, permet d'échantillonner selon la méthode de répulsion adaptative
- repul_param initialement à 0, permet d'imposer un écartement minimal entre les points dans chaque nuage de l'échantillon

Si jln_mth et inert_pena_ch sont tous deux à False et repul_param est à 0, alors l'échantillon obtenu en sortie de l'algorithme est celui obtenu avec un maximin sans pénalisation.

Pour chacune de ces méthodes, il est possible de choisir une initialisation par LHS ou une initialisation par un simple tirage uniforme. Cependant, nous préconisons d'utiliser le LHS uniquement pour la méthode jln ou l'on observe une convergence plus rapide en moyenne, là où l'on observe plutôt le contraire avec les méthodes basé sur le maximin. Mais quelle que soit l'initialisation, les méthodes convergent et ressortent des résultats satisfaisants.


Il est également possible d'obtenir un échantillon avec la méthode par LHS, qui est utilisée pour l'initialisation de certaines de nos méthodes. Pour ceci, il suffit d'utiliser le code suivant :

```python
import initialisation

P_list_f, _ = initialisation.initialisation(Nmin, Nmax, nb_sample, born_inf, born_sup, d
                 all_opt = False, aff = False, for_torch = False, lhs = True)
```
Avec les 6 premières variables ayant la même fonction que dans l'exemple d'exécutions précèdent et les variable suivante ayant pour but de renvoyer le nuage de points dans le même forma que la sortie de l'autre code.


## Exemple d'appel
Le code suivant permet ainsi d'obtenir les résultats avec la méthode avec répulsion adaptative.


```python
import exmple_appel

P_list_f = creation_dun_ech(Nmin = 28, Nmax = 35, nech = 100, jln_mth = False, d = 2, export_all=True, born_inf = [0,0], born_sup=[1,1],
inert_pena_ch = True, repul_param = 0, aff = True, seed = 12345678, lhs = False, save = True)
```

Les sorties que vous devriez obtenir sont dans le dossier résultat_exemple de ce dépôt.

Pour toute précision sur les méthodes d'échantillonnage, nous vous prions de vous référer au rapport également présent dans ce dépôt. Toutefois, si des questions restent en suspens, vous pouvez me joindre sur l'adresse mail loick.diener@gmail.com .










