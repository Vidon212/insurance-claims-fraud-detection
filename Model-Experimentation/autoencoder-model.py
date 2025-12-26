import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.tensorflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Health Insurance Claim Anomaly Detection")

@mlflow.trace(name="load_and_preprocess_data")
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    features = ['claim_amount', 'num_services', 'patient_age', 'provider_id', 'days_since_last_claim']
    
    # Standardize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[features])
    
    # Split the data into training and test sets
    X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)
    return X_train, X_test, scaler

@mlflow.trace(name="build_autoencoder")
def build_autoencoder(input_dim):
    # Input layer
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    encoded = Dense(32, activation='relu')(input_layer)
    encoded = Dense(16, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(32, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded) # Using sigmoid for normalized data if typical, but here linear/relu might be better if not 0-1. 
    # Since we used StandardScaler, values are varying. Linear activation for output is safer, or we rethink normalization.
    # Let's stick to simple Dense default (linear) for output reconstruction of StandardScaled data.
    output_layer = Dense(input_dim)(decoded)

    autoencoder = Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

@mlflow.trace(name="train_autoencoder")
def train_autoencoder(model, X_train, epochs=50, batch_size=32):
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.1,
        verbose=1
    )
    return history

@mlflow.trace(name="detect_anomalies")
def detect_anomalies(model, X, threshold=None):
    reconstructions = model.predict(X)
    mse = np.mean(np.power(X - reconstructions, 2), axis=1)
    
    if threshold is None:
        # If no threshold is provided, use a statistical one (e.g., mean + 2*std)
        threshold = np.mean(mse) + 2 * np.std(mse)
    
    anomalies = mse > threshold
    return anomalies.astype(int), mse, threshold

def run_pipeline():
    with mlflow.start_run():
        # Load and preprocess data
        X_train, X_test, scaler = load_and_preprocess_data('../Data-Generation/random_health_claims.csv')
        
        input_dim = X_train.shape[1]
        
        # Log parameters
        epochs = 50
        batch_size = 32
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("model_type", "Autoencoder")

        # Build and train model
        autoencoder = build_autoencoder(input_dim)
        history = train_autoencoder(autoencoder, X_train, epochs=epochs, batch_size=batch_size)
        
        # Calculate threshold on training data to determine what is "normal" reconstruction error
        _, mse_train, threshold = detect_anomalies(autoencoder, X_train)
        mlflow.log_param("threshold", threshold)

        # Predict on test set
        y_pred_test, mse_test, _ = detect_anomalies(autoencoder, X_test, threshold=threshold)
        
        # Log metrics
        test_anomaly_percentage = y_pred_test.mean() * 100
        mlflow.log_metric("test_anomaly_percentage", test_anomaly_percentage)
        mlflow.log_metric("final_train_loss", history.history['loss'][-1])
        mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])

        # Log the model
        mlflow.tensorflow.log_model(autoencoder, artifact_path="model")

        print(f"Test Anomaly Percentage: {test_anomaly_percentage:.2f}%")
        print(f"Threshold: {threshold}")
        print("Model, metrics, and traces logged to MLflow.")

if __name__ == "__main__":
    run_pipeline()
