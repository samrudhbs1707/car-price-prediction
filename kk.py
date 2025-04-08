import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv('car data.csv')

# Preprocess data
le = LabelEncoder()
data['Fuel_Type'] = le.fit_transform(data['Fuel_Type'])
data['Seller_Type'] = le.fit_transform(data['Seller_Type'])
data['Transmission'] = le.fit_transform(data['Transmission'])
data = data.drop(['Car_Name'], axis=1)

X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save model
with open('car_price_model.pkl', 'wb') as f:
    pickle.dump((model, scaler, le), f)

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    return render_template('index.html', prediction_text=f'Predicted Price: {prediction:.2f} Lakhs')

if __name__ == '__main__':
    app.run(debug=True)
