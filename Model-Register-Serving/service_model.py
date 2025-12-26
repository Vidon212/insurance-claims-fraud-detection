import bentoml
import numpy as np
import pandas as pd

# Load the registered model using the new BentoML API
bento_model = bentoml.sklearn.get("health_insurance_anomaly_detector:latest")

@bentoml.service(
    name="health_insurance_anomaly_detection_service",
    resources={"cpu": "1"}
)
class AnomalyDetectionService:
    def __init__(self):
        # Directly load the model into memory
        self.model = bento_model.load_model()

    @bentoml.api
    def predict(self, data: pd.DataFrame) -> dict:
        # Make predictions
        predictions = self.model.predict(data)
        return {"predictions": predictions.tolist()}
