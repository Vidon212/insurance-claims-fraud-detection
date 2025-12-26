import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Health Insurance Claim Anomaly Detection")

@mlflow.trace(name="load_data")
def load_data(filepath):
    df = pd.read_csv(filepath)
    features = ['claim_amount', 'num_services', 'patient_age', 'provider_id', 'days_since_last_claim']
    split_data = train_test_split(df[features], test_size=0.2, random_state=42)
    return split_data

@mlflow.trace(name="train_model")
def train_model(X_train, n_estimators=100, contamination=0.05):
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    model.fit(X_train)
    return model

@mlflow.trace(name="predict_anomalies")
def predict_anomalies(model, X):
    return model.predict(X)

def run_pipeline():
    with mlflow.start_run():
        # Load and split data
        X_train, X_test = load_data('../Data-Generation/random_health_claims.csv')

        # Log parameters
        n_estimators = 100
        contamination = 0.05
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("contamination", contamination)

        # Train model
        model = train_model(X_train, n_estimators=n_estimators, contamination=contamination)

        # Predict
        y_pred_train = predict_anomalies(model, X_train)
        y_pred_test = predict_anomalies(model, X_test)
        
        # Convert predictions to anomaly scores (-1 is anomaly, 1 is normal)
        anomaly_score_train = (y_pred_train == -1).astype(int)
        anomaly_score_test = (y_pred_test == -1).astype(int)

        # Log metrics
        train_anomaly_percentage = anomaly_score_train.mean() * 100
        test_anomaly_percentage = anomaly_score_test.mean() * 100
        
        mlflow.log_metric("train_anomaly_percentage", train_anomaly_percentage)
        mlflow.log_metric("test_anomaly_percentage", test_anomaly_percentage)

        # Log the model (using artifact_path to be explicit)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Train Anomaly Percentage: {train_anomaly_percentage:.2f}%")
        print(f"Test Anomaly Percentage: {test_anomaly_percentage:.2f}%")
        print("Model, metrics, and traces logged to MLflow.")

if __name__ == "__main__":
    run_pipeline()
