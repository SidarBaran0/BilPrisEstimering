import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def train_model(data_path):
    # Laste inn datasettet med riktig skilletegn
    data = pd.read_csv(data_path, sep=';')

    # Forhåndsvis data for feilsøking
    print("Kolonner i datasettet:", data.columns.tolist())
    print(data.head())

    # Forbehandle data (konverter kategoriske kolonner til numeriske)
    data = pd.get_dummies(data, columns=['Brand', 'Model', 'Fuel_Type', 'Transmission'], drop_first=True)

    # Split funksjoner (X) og målvariabel (y)
    X = data.drop('Price', axis=1)  # Fjern målvariabelen fra funksjonene
    y = data['Price']  # Målvariabel

    # Splitt data i trenings- og testsett
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tren Random Forest-modellen
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluer modellen
    y_pred = model.predict(X_test)
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

    # Lagre modellen
    joblib.dump(model, 'models/car_price_model.pkl')
