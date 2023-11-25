from flask import Flask, request, jsonify, render_template
import joblib
from preprocesamiento import preprocesar_datos
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__) 
print('Loading model and preprocessors')
model = joblib.load("/home/ubuntu/carpeta/api_DSA/model_assets/modelo_random_forest.pkl")
ohe = joblib.load("/home/ubuntu/carpeta/api_DSA/model_assets/ohe.pkl")
mmscaler = joblib.load("/home/ubuntu/carpeta/api_DSA/model_assets/mmscaler.pkl")

@app.route('/predict', methods=['POST']) 
def predict(): 
    try:
        data = request.get_json()
        SMOKE = data['SMOKE']
        CALC = data['CALC']
        NCP = data['NCP']
        CH2O = data['CH2O']
        TUE = data['TUE']
        MTRANS = data['MTRANS']
        FCVC = data['FCVC']
        FAVC = data['FAVC']

        print('Readed data from request')

        data_dict = {'SMOKE': [SMOKE], 'CALC': [CALC], 'NCP': [NCP], 'CH2O': [CH2O], 'TUE': [TUE], 'MTRANS': [MTRANS], 'FCVC': [FCVC], 'FAVC': [FAVC]}
        df = pd.DataFrame(data=data_dict)
        df_processed = preprocesar_datos(df, ohe, mmscaler)
        
        prediction = model.predict(df_processed) 
        return jsonify(prediction.tolist())  # Convierte el resultado a lista antes de jsonify

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/')
def index():
    return render_template('index.html')
    #return render_template('C:/Users/Juan/Downloads/api final final/templates/website.html')

# Ejecutar la aplicación 
if __name__ == '__main__':
    app.run(debug=True)
