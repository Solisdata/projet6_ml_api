- Readme
    
    # Projet 6 – API Prédiction Consommation Énergétique


    ## 1. Objectif
    
    Prédire la consommation énergétique d’un bâtiment à partir de ses caractéristiques (type, localisation, année de construction, surface, nombre d’étages…).
    
    Créer un API sur le cloud (AWS) permettant d’intérroger le modèle de prédiction.



    ## 2. Technologies utilisées
    Python   
    Docker  
    BentoML
    Poetry
    Artifact Registry (GCP)
    Cloud Run (GCP)


    ## 3. Contenu du projet
  ```
   project/
    ├── docs/
    │   └── OCR_projet6_mathilde_lesolliec_présentation.pdf   # Présentation du projet
    │
    ├── data/
    │   ├── 2016_Building_Energy_Benchmarking.csv     # Jeu de données source
    │   └── building_clean.csv                        # Données nettoyées après preprocessing
    │
    ├── modele/
    │   ├── 1_analyse_exploratoire.ipynb              # Analyse exploratoire des données (EDA)
    │   ├── 2_modelisation_supervisée.ipynb           # Modélisation supervisée
    │   
    ├── service.py                                     #  Service - API
    │
    ├──.gitignore                                      # Fichiers exclus de Git
    │
    └── setup/
        ├── bentofile.yaml                             # Configuration BentoML pour le déploiement
        ├── pyproject.toml                             # Gestion des dépendances (Poetry)
        └── poetry.lock                                # Versions figées des dépendances
     ```


    ## 4. Machine Learning
    
    - **Modèle utilisé** : Gradient Boosting (scikit-learn)
    - **Données** : caractéristiques des bâtiments + consommation énergétique
    - **Étapes** :
        1. Analyse et préparation les données (features et target)
        2. Entraînement du modèle 
        3. Évaluation de  la performance et amélioration
        4. Sauvegarder le modèle dans le store BentoML
    
    ## 
    
    ```python
    import bentoml
    bentoml.sklearn.save_model("bes_model", best_gb)
    
    ```


    ## 5. Création de l’API et déploiement sur le CLOUD
    
    - Création de l’API avec le fichier **service.py**
    - Création d’une image docker
    - **Lancer l’API avec Docker :**
        
        ```bash
        bentoml build
        
        ```
        
    - **Tester les endpoints  avec Swagger**
        - Exemple de donner à rentrer : Prédiction :
            
            ```powershell
            {
                input_data = @{
                    PrimaryPropertyType = "Office"
                    Latitude = 47.6
                    Longitude = -122.3
                    YearBuilt = 2000
                    NumberofFloors = 5
                    PropertyGFABuilding_s = 10000
                    pct_electricity = 0.7
                    pct_steam = 0.3
                }
            } 
            
            ```
            
    
    ## 6. Déploiement Cloud (GCR)
    
    1. Tag et push de l’image Docker
    2. Déploiement sur Cloud Run pour rendre l’API publique.








