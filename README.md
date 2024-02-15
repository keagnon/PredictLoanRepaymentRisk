# Modèle de prédiction du risque de remboursement de crédit bancaire

Ce projet vise à développer un modèle de machine learning pour prédire le risque de remboursement de crédit bancaire, permettant ainsi aux institutions financières de prendre des décisions plus éclairées en matière de prêts.

## Objectifs

- Construire un modèle de prédiction du risque de remboursement de crédit en utilisant des techniques de machine learning.
- Classifier les clients en fonction de leur probabilité de rembourser leur crédit.
- Utiliser Google Cloud Platform (GCP) pour l'entraînement et le déploiement du modèle.

## Technologies utilisées

- **Langage de programmation** : Python
- **Librairies ML** : scikit-learn
- **Plateforme Cloud** : Google Cloud Platform (GCP)
- **Outils de développement** : Jupyter Lab
- **Gestion de versions** : Git

## Structure du projet

```
|- data/                    # Répertoire pour les données
|   |- credit_data.csv      # Fichier CSV contenant les données du crédit
|- notebooks/               # Répertoire pour les notebooks Jupyter
|   |- Credit_Risk_Prediction.ipynb    # Notebook contenant l'analyse des données et la construction du modèle
|- models/                  # Répertoire pour sauvegarder les modèles entraînés
|- screenshots/             # Répertoire pour les captures d'écran
|   |- GCP_Workbench.png    # Capture d'écran de l'interface Workbench sur GCP
|   |- JupyterLab_GCP.png   # Capture d'écran du notebook dans Jupyter Lab sur GCP
|- README.md                # Documentation principale du projet
```

## Instructions d'utilisation

1. **Installation des dépendances** :

   Assurez-vous d'avoir installé toutes les dépendances nécessaires répertoriées dans le fichier `requirements.txt`. Vous pouvez les installer en exécutant la commande suivante :
   ```
   pip install -r requirements.txt
   ```

2. **Exécution du notebook** :

   Ouvrez le notebook `Credit_Risk_Prediction.ipynb` dans Jupyter Lab. Exécutez chaque cellule pour analyser les données, entraîner le modèle et évaluer les performances.

3. **Déploiement sur GCP** :

   Utilisez l'interface Workbench de GCP pour charger les données, entraîner le modèle et le déployer en production. Référez-vous à la documentation de GCP pour plus de détails sur ces étapes.

## Captures d'écran

- **Interface Workbench sur GCP** :
  ![GCP Workbench](screenshots/GCP_Workbench.png)

- **Notebook dans Jupyter Lab sur GCP** :
  ![JupyterLab GCP](screenshots/JupyterLab_GCP.png)

## Auteur

GBE Keagnon Grâce Helena

