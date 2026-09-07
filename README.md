# Sampling_point_cloud

Sampling_point_cloud est un ensemble de méthode d'échantillonnage sur l'ensemble des nuage de points.

## Installation
Avant de pouvoir lancer les différent script python, il est necessaire de set-up un environnement conda en utilisant les commandes suivante :

```bash
conda create -n your_env_name python=3.12 numpy=2.5.2 matplotlib=3.11.1 pytorch=2.6.0 bioconda::geomloss=0.3.1
conda activate your_env_name
```
Avec your_env_name le nom de l'environnement que vous voulez créer. 

La commande ci-dessus installe python 3.12 dans l'environnement mais si vous avez une quelconque autre préférence ces scripte fonctionne avec les version de python de 3.10 à 3.14.3. Pour ce qui est d'autre version le teste n'a pas été effectué.

## Validation

Pour confirmer que l'ensemble des libraris soit correctement installé nous vous conseillons d'effectuer la commande :

```bash
python ./exmple_appel.py
```
Les nuage de points générer par cette apppel seront alors présent dans le dossié resultat_optim nouvellement créé
