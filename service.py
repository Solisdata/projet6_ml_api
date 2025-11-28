import bentoml
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# Définir le schéma d'entrée
class InputData(BaseModel):
    PrimaryPropertyType: str = Field(description="Type de propriété primaire")
    Latitude: float = Field(ge=47.0, le=48.0, description="Latitude du bâtiment (entre 47.0 et 48.0)") 
    Longitude: float = Field(ge=-123.0, le=-122.0, description="Longitude du bâtiment (entre -123.0 et -122.0)")
    YearBuilt: int = Field(ge=1800, le=2016, description="Année de construction (doit être réaliste)")
    NumberofFloors: int = Field(gt=0, le=100, description="Nombre d'étages (doit être > 0 et <= 100)")
    PropertyGFABuilding_s: int = Field(gt=0, description="Surface du bâtiment en pieds carrés (doit être > 0)")
    pct_electricity: float = Field(ge=0.0, le=1.0, description="Pourcentage d'électricité de la conso. totale [0, 1]")
    pct_steam: float = Field(ge=0.0, le=1.0, description="Pourcentage de vapeur de la conso. totale [0, 1]")

# Charger le modèle
model_ref = bentoml.sklearn.get("best_model:latest")

# types de bâtiments vus à l'entraînement
KNOWN_BUILDING_TYPES = [
    "Healthcare", "Hospitality", "Office", "Other", 
    "Retail", "Warehouse", "Worship Facility"
]

@bentoml.service()
class EnergyAPI_test:
    
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
            
            result_log = model.predict(df)
            result_real = np.expm1(result_log[0])
            return {"prediction_kBtu": float(result_real)}
        
        except Exception as e:
            return {"error": str(e), "type": str(type(e))}

# Pour le lancer : bentoml serve service:EnergyAPI_test --reload

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