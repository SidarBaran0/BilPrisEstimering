from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Last inn modellen
model = joblib.load('../models/car_price_model.pkl')

# Opprett prediksjonsendepunkt
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Hent JSON-data fra forespørselen
    df = pd.DataFrame(data, index=[0])  # Konverter til DataFrame
    prediction = model.predict(df)  # Utfør prediksjon
    return jsonify({'price': prediction[0]})  # Returner predikert pris

if __name__ == '__main__':
    app.run(debug=True)  # Start Flask-serveren
