import bentoml
from pydantic import BaseModel
import pandas as pd

# Définir le schéma d'entrée
class InputData(BaseModel):
    PrimaryPropertyType: str
    Latitude: float
    Longitude: float
    YearBuilt: int
    NumberofFloors: int
    PropertyGFAParking: int
    PropertyGFABuilding_s: int
    pct_electricity: float
    pct_steam: float
    has_parking: bool

# Charger le modèle
model_ref = bentoml.sklearn.get("best_rf_model:latest")

@bentoml.service()
class EnergyAPI:
    
    @bentoml.api
    def home(self) -> dict:
        """
        Endpoint d'accueil de l'API
        """
        return {
            "message": "Bienvenue sur l'API de prédiction de consommation énergétique !",
            "description": "Cette API permet de prédire la consommation énergétique d'un bâtiment.",
            "endpoints": {
                "/": "Page d'accueil (GET)",
                "/predict": "Endpoint de prédiction (POST)"
            },
            "exemple_requete": {
                "PrimaryPropertyType": "Office",
                "Latitude": 47.6,
                "Longitude": -122.3,
                "YearBuilt": 2000,
                "NumberofFloors": 5,
                "PropertyGFAParking": 1000,
                "PropertyGFABuilding_s": 10000,
                "pct_electricity": 0.7,
                "pct_steam": 0.3,
                "has_parking": True
            }
        }
    
    @bentoml.api
    def predict(self, input_data: InputData) -> dict:
        """
        Endpoint de prédiction de consommation énergétique
        """
        # Charger le modèle
        model = model_ref.load_model()
        
        # Conversion en DataFrame
        df = pd.DataFrame([input_data.model_dump()])
        
        # Convertir le booléen en int si nécessaire
        df['has_parking'] = df['has_parking'].astype(int)
        
        # Prédiction
        result = model.predict(df)
        
        return {"prediction": float(result[0])}

# Pour le lancer : bentoml serve service:EnergyAPI --reload

# Pour tester :
# documentation Swagger** dans le navigateur  : http://localhost:3000
# GET home : curl http://localhost:3000/home
# 
# POST predict : 
# curl -X POST http://localhost:3000/predict \
#   -H "Content-Type: application/json" \
#   -d '{
#     "PrimaryPropertyType": "Office",
#     "Latitude": 47.6,
#     "Longitude": -122.3,
#     "YearBuilt": 2000,
#     "NumberofFloors": 5,
#     "PropertyGFAParking": 1000,
#     "PropertyGFABuilding_s": 10000,
#     "pct_electricity": 0.7,
#     "pct_steam": 0.3,
#     "has_parking": true
#   }'