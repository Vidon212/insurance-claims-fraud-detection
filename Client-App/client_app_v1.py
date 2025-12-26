from flask import Flask, render_template, request
import pandas as pd
import requests

app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the CSV file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400

    if file:
        # Read the CSV file directly from the file object
        df = pd.read_csv(file)

        # Separate the 'claim_id' column if it exists
        if 'claim_id' in df.columns:
            claim_ids = df['claim_id']
            df_features = df.drop(columns=['claim_id'])
        else:
            claim_ids = None
            df_features = df

        # Send the DataFrame to the BentoML service
        # Using orient='split' as established in test_claims.py
        response = requests.post(
            'http://127.0.0.1:3000/predict',  # BentoML endpoint
            json=df_features.to_dict(orient='split')
        )
        
        if response.status_code != 200:
            return f"Error from prediction service: {response.text}", 500

    # Get predictions from the response
    predictions = response.json()['predictions']

    # Add predictions to the DataFrame
    df['Prediction'] = predictions

    # Reattach the 'claim_id' column to the DataFrame
    if claim_ids is not None:
        df['claim_id'] = claim_ids

    # Reorder columns to have 'claim_id' first
    if 'claim_id' in df.columns:
        df = df[['claim_id'] + [col for col in df.columns if col != 'claim_id']]

    # Render the DataFrame as an HTML table
    return render_template('result.html', tables=[df.to_html(classes='data', header="true")])

if __name__ == '__main__':
    app.run(debug=True, port=5005)
