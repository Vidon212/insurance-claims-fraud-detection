from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import requests
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


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

        # Separate the 'claim_id' column for feature preparation if it exists
        if 'claim_id' in df.columns:
            claim_ids = df['claim_id']
            df_features = df.drop(columns=['claim_id'])
        else:
            claim_ids = None
            df_features = df

        # Send the DataFrame to the BentoML service
        # Using orient='split' for BentoML pandas support
        response = requests.post(
            'http://127.0.0.1:3000/predict',
            json=df_features.to_dict(orient='split')
        )
        
        if response.status_code != 200:
            return f"Error from prediction service: {response.text}", 500

        # Get predictions from the response
        predictions = response.json()['predictions']

        # Add predictions to the ORIGINAL DataFrame so we keep claim_id + features + prediction
        df['Prediction'] = predictions

        # Save the DataFrame to a session file for visualization
        df.to_csv('session_data.csv', index=False)

        # Render the DataFrame as an HTML table with a link to visualize
        return render_template('result.html', tables=[df.to_html(classes='data', header="true")])

# Route to handle the visualization
@app.route('/visualize')
def visualize():
    # Load the session data
    df = pd.read_csv('session_data.csv')

    # Create a pie chart based on the 'Prediction' column
    prediction_counts = df['Prediction'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(prediction_counts, labels=prediction_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Prediction Distribution')

    # Save the pie chart as an image
    if not os.path.exists('static'):
        os.makedirs('static')
    chart_path = 'static/prediction_pie_chart.png'
    plt.savefig(chart_path)
    plt.close()

    # Render the visualization page with the pie chart
    return render_template('visualize.html', chart_path=chart_path)

if __name__ == '__main__':
    app.run(debug=True, port=5005)
