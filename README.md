# Insurance Claims Fraud Detection

Health Insurance Claims Fraud Detection Project.

## Project Structure
- `Data-Generation/`: Contains scripts for generating synthetic health claims data.
- `Model-Experimentation/`: Scripts for training models and tracking experiments with MLflow.

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
   python3 isolation-forest-model.py
   ```
   This script trains an Isolation Forest model, logs metrics, params, and traces to MLflow.

3. Run the Autoencoder model experiment:
   ```bash
   python3 autoencoder-model.py
   ```
   This script trains an Autoencoder neural network to detect anomalies based on reconstruction error.

## Next Steps
- Deploy the best performing model.
