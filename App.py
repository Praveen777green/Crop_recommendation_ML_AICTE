from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib

app = Flask(__name__)

# Load necessary data and models
crop_dict = {
    0: 'rice', 1: 'maize', 2: 'chickpea', 3: 'kidneybeans', 4: 'pigeonpeas', 5: 'mothbeans', 
    6: 'mungbean', 7: 'blackgram', 8: 'lentil', 9: 'pomegranate', 10: 'banana', 11: 'mango', 
    12: 'grapes', 13: 'watermelon', 14: 'muskmelon', 15: 'apple', 16: 'orange', 17: 'papaya', 
    18: 'coconut', 19: 'cotton', 20: 'jute', 21: 'coffee'
}

# Load trained model (Save the model after training)
# Assuming you've already trained the model and saved it
# dtc = DecisionTreeClassifier()
# dtc.fit(x_train_scaled, y_train)
# joblib.dump(dtc, 'crop_model.pkl')
# joblib.dump(scaler, 'scaler.pkl')

# Load model and scaler
dtc = joblib.load('crop_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop', methods=['POST'])
def crop():
    try:
        N = int(request.form['N'])
        P = int(request.form['P'])
        K = int(request.form['K'])
        temp = float(request.form['temp'])
        hum = float(request.form['hum'])
        ph = float(request.form['ph'])
        rain = float(request.form['rain'])
        
        # Prepare the input data for prediction
        features = np.array([[N, P, K, temp, hum, ph, rain]])
        transformed_features = scaler.transform(features)
        
        # Predict the crop
        prediction = dtc.predict(transformed_features)
        crop_name = crop_dict[prediction[0]]
        
        return render_template('result.html', crop=crop_name)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
