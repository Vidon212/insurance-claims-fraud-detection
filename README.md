# Insurance Claims Fraud Detection

Health Insurance Claims Fraud Detection Project.

## Project Structure
- `Data-Generation/`: Contains scripts for generating synthetic health claims data.
- `Model-Experimentation/`: Scripts for training models and tracking experiments with MLflow.
- `Model-Register-Serving/`: Scripts for registering and serving models using BentoML.
- `Client-App/`: Flask application for end-user interaction.

## Setup
1. Create and activate a virtual environment:
   ```bash
   python3 -m venv mlops-env
   source mlops-env/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Generation
Run the random health claims generator:
```bash
cd Data-Generation
python3 random_health_claims.py
```
This will create a `random_health_claims.csv` file with synthetic data including normal claims and injected anomalies.

## Model Experimentation
1. Start the MLflow UI in a separate terminal:
   ```bash
   mlflow ui
   ```
   Open http://127.0.0.1:5000 in your browser.

2. Run the Isolation Forest model experiment:
   ```bash
   cd Model-Experimentation
   python3 isolation_forest_model.py
   ```
   This script trains an Isolation Forest model, logs metrics, params, and traces to MLflow.

3. Run the Autoencoder model experiment:
   ```bash
   python3 autoencoder_model.py
   ```
   This script trains an Autoencoder neural network to detect anomalies based on reconstruction error.

## Model Registration
1. Register the trained model with BentoML:
   ```bash
   cd Model-Register-Serving
   python3 register_model.py
   ```
   This loads the trained model (expected at `../model.pkl`) and saves it to the BentoML model store.

2. Verify registration:
   ```bash
   bentoml models list
   ```

## Model Serving
1. Serve the model locally:
   ```bash
   cd Model-Register-Serving
   bentoml serve service_model.py:AnomalyDetectionService --reload
   ```
   This starts a local API server at http://localhost:3000.

2. Test the endpoint (example using curl):
   ```bash
   curl -X POST -H "Content-Type: application/json" \
        -d '{"data": [[1000, 2, 45, 10, 30]]}' \
        http://localhost:3000/predict
   ```

3. Test the endpoint (using test_claims.py):
   ```bash
   cd Model-Register-Serving
   python3 test_claims.py
   ```
   **Response Interpretation:**
   - `1`: Claim approved (Normal)
   - `-1`: Claim rejected (Anomaly detected)

## Client App

The client app provides a web interface to upload claims data and visualize fraud detection results. There are two versions available:

### Version 1: Basic
This version displays the fraud predictions in a simple table format.
1. Run the app:
   ```bash
   cd Client-App
   python3 client_app_v1.py
   ```
2. Visit http://127.0.0.1:5005 to upload `random_health_claims.csv`.

### Version 2: Advanced (Visualization)
This version includes graphical visualizations of the prediction distribution.
1. Run the app:
   ```bash
   cd Client-App
   python3 client_app_v2.py
   ```
2. Visit http://127.0.0.1:5005 to upload `random_health_claims.csv`.
3. After results appear, visit http://127.0.0.1:5005/visualize to view the prediction distribution charts.

**Note:** Stop one version (`Ctrl+C`) before running the other, as they both use the same port (5005). 
   
   
