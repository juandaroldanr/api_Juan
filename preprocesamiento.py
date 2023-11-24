import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocesar_datos(df):
    # Conversión de variables categóricas binarias a unos y ceros
    df['SMOKE'] = df['SMOKE'].map({'yes': 1, 'no': 0})
    
    # Obtener columnas de diferentes tipos
    catcols = df.select_dtypes(exclude=['int64', 'float64']).columns
    intcols = df.select_dtypes(include=['int64']).columns
    floatcols = df.select_dtypes(include=['float64']).columns
    
    # One-hot encoding para variables categóricas
    df = pd.get_dummies(df, columns=catcols)
    
    # MinMax scaling para variables numéricas
    scaler = MinMaxScaler()
    df[floatcols] = scaler.fit_transform(df[floatcols])
    df[intcols] = scaler.fit_transform(df[intcols])
    
    return df
