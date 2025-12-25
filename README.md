# Insurance Claims Fraud Detection

Health Insurance Claims Fraud Detection Project.

## Project Structure
- `Data-Generation/`: Contains scripts for generating synthetic health claims data.
- `requirements.txt`: Python dependencies.

## Setup
1. Install dependencies:
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

## Next Steps
- Model Development
- Experimenting with various algorithms for fraud detection
