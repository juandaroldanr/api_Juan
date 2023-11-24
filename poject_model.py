#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10

@authors: Diego Garcia, Jairo Rueda, Laura Camila Peralta, Juan Roldán 
"""

# Importe el conjunto de datos de diabetes y divídalo en entrenamiento y prueba usando scikit-learn
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.preprocessing import OneHotEncoder

#Se cargan los datos.

#df = pd.read_csv('C:\Users\Juan\Downloads\ObesityDataSet_raw_and_data_sinthetic.csv')
df = pd.read_csv(r'C:\Users\Juan\Downloads\ObesityDataSet_raw_and_data_sinthetic.csv')


#Variables a conservar : SMOKE si fuma o no, CALC cantidad de alchool que consume, NCP comidas al día, CH2O litros agua al día,FAF qué tan seguido se ejercita, TUE tiempo usando dispositivos electrónicos, MTRANS medio de transporte, FCVC frecuencia consumo vegetales 
df = df.drop(columns=['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'CAEC', 'SCC', 'FAF'])

#preprocesamiento

#Se separa la acoluma a predecir en el modelo de machine learning y también se vuelve binaria la variable de respuesta
y = df["NObeyesdad"]
df = df.drop(columns=['NObeyesdad'])
#y = np.where((y == 'Normal_Weight') | (y == 'Insufficient_Weight'), 0, 1)


#Conversión de variables categóricas binarias a unos y ceros

df['SMOKE'] = df['SMOKE'].map({'yes': 1, 'no': 0})

catcols = df.select_dtypes(exclude = ['int64','float64']).columns
intcols = df.select_dtypes(include = ['int64']).columns
floatcols = df.select_dtypes(include = ['float64']).columns

# one-hot encoding para variables categóricas
#df = pd.get_dummies(df, columns = catcols)
ohe = OneHotEncoder(sparse = False)
#df = ohe.fit(df, columns = catcols)

#df = ohe.fit(df[catcols])
#df[catcols] = ohe.fit_transform(df[catcols])
# one-hot encoding para variables categóricas
df_encoded = pd.DataFrame(ohe.transform(df[catcols]), columns=ohe.get_feature_names(catcols))
df = pd.concat([df.drop(columns=catcols), df_encoded], axis=1)

#df[catcols] = ohe.fit_transform(df[catcols])
ruta_ohe = r'C:\Users\Juan\Downloads\ohe.joblib'
joblib.dump(ohe, ruta_ohe)


# minmax scaling para variabls numéricas
#for col in df[floatcols]:
#    df[col] = MinMaxScaler().fit_transform(df[[col]])

#for col in df[intcols]:
#    df[col] = MinMaxScaler().fit_transform(df[[col]])


scaler = MinMaxScaler()
df[floatcols + intcols] = scaler.fit_transform(df[floatcols + intcols])



#scaler = MinMaxScaler()
#scaler.fit()
ruta_scaler = r'C:\Users\Juan\Downloads\scaler.joblib'
joblib.dump(scaler, ruta_scaler)



    
# Se dividen los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

from sklearn.metrics import mean_squared_error

n_estimators = 100 
max_depth = 6
max_features = 4

modelo_random_forest = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
modelo_random_forest.fit(X_train, y_train) 
predictions = modelo_random_forest.predict_proba(X_test)
exactitud = accuracy_score(y_test, modelo_random_forest.predict(X_test))
print(f'Exactitud: {exactitud:.2f}')

ruta_modelo = r'C:\Users\Juan\Downloads\random_forest_model.joblib'
joblib.dump(modelo_random_forest, ruta_modelo)
print(f"Modelo guardado en: {ruta_modelo}")



