import pandas as pd


def preprocess_data(file_path):
    # Les inn datasettet
    data = pd.read_csv(file_path)

    # Fyll inn manglende verdier med medianen
    data.fillna(data.median(), inplace=True)

    # Konverter kategoriske kolonner til numeriske
    data = pd.get_dummies(data, columns=['Brand', 'Fuel_Type', 'Transmission'], drop_first=True)

    return data
