from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model from the current directory
# Make sure model.pkl is in the same folder as app.py on your AWS instance
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: model.pkl not found. Please ensure it is uploaded to the server.")
    model = None

# The exact feature names extracted from your pickle file
EXPECTED_FEATURES = [
    'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]

@app.route('/health', methods=['GET'])
def health_check():
    """AWS Load Balancers use this to check if the API is running."""
    if model is None:
        return jsonify({'status': 'unhealthy', 'reason': 'Model not loaded'}), 500
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not available on the server.'}), 500

    try:
        data = request.get_json()
        
        # Ensure all required features are present
        input_values = []
        for feature in EXPECTED_FEATURES:
            if feature not in data:
                return jsonify({'error': f'Missing required feature: {feature}'}), 400
            input_values.append(data[feature])
            
        # Convert to a pandas DataFrame so scikit-learn receives the feature names
        # This prevents the "X does not have valid feature names" warning
        df = pd.DataFrame([input_values], columns=EXPECTED_FEATURES)
        
        # Generate prediction and probability
        prediction = model.predict(df)
        probability = model.predict_proba(df).tolist()[0]
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probabilities': {
                'class_0': probability[0],
                'class_1': probability[1]
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app locally for testing
    app.run(host='0.0.0.0', port=5000)
