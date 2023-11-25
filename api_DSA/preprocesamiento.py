import pandas as pd
import numpy as np

def preprocesar_datos(df, onehotenc, minmaxsc):
    # Conversión de variables categóricas binarias a unos y ceros
    df['SMOKE'] = df['SMOKE'].map({'yes': 1, 'no': 0})
    df['FAVC'] = df['FAVC'].map({'yes': 1, 'no': 0})
    
    # Obtener columnas de diferentes tipos
    catcols = df[['FAVC','SMOKE','CALC','MTRANS']]
    numcols = df[['FCVC','NCP','CH2O','TUE']]
    
    # One-hot encoding para variables categóricas
    print("Transforming categorical columns")
    catcols_encoded = onehotenc.transform(catcols)
    print("Transforming numerical columns")
    numcols_scaled = minmaxsc.transform(numcols)
    df_processed = np.concatenate([catcols_encoded, numcols_scaled], axis = 1)
    return df_processed
