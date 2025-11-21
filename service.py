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
    PropertyGFABuilding_s: int
    pct_electricity: float
    pct_steam: float

# Charger le modèle
model_ref = bentoml.sklearn.get("best_rf_model:latest")

# types de bâtiments vus à l'entraînement
KNOWN_BUILDING_TYPES = [
    "Healthcare", "Hospitality", "Office", "Other", 
    "Retail", "Warehouse", "Worship Facility"
]

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
                "PropertyGFABuilding_s": 10000,
                "pct_electricity": 0.7,
                "pct_steam": 0.3
            }
        }
    
    @bentoml.api
    def predict(self, input_data: InputData) -> dict:
        try:
            model = model_ref.load_model()
            df = pd.DataFrame([input_data.model_dump()])
            
            for b_type in KNOWN_BUILDING_TYPES:
                df[f'PrimaryPropertyType_{b_type}'] = (df['PrimaryPropertyType'] == b_type).astype(int)
            df = df.drop(columns=['PrimaryPropertyType'])
            
            result = model.predict(df)
            return {"prediction": float(result[0])}
        
        except Exception as e:
            return {"error": str(e), "type": str(type(e))}

# Pour le lancer : bentoml serve service:EnergyAPI --reload

# Pour tester :
# documentation Swagger** dans le navigateur  : http://localhost:3000
# GET home : curl http://localhost:3000/home
# 
# POST predict : 
# curl -X POST http://localhost:3000/predict \
#   -H "Content-Type: application/json" \
#   -d '{

"""
{
  "input_data": {
    "PrimaryPropertyType": "Office",
    "Latitude": 47.6,
    "Longitude": -122.3,
    "YearBuilt": 2000,
    "NumberofFloors": 5,
    "PropertyGFABuilding_s": 10000,
    "pct_electricity": 0.7,
    "pct_steam": 0.3
  }
}
"""