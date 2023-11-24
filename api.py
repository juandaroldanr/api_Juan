from flask import Flask, request, jsonify, render_template
import joblib
from preprocesamiento import preprocesar_datos
import pandas as pd

app = Flask(__name__) 
model = joblib.load("C:/Users/Juan/Downloads/random_forest_model_with_preprocessing.joblib")
ohe = joblib.load("C:/Users/Juan/Downloads/random_forest_model_with_preprocessing.joblib")

@app.route('/predict', methods=['POST']) 
def predict(): 
    try:
        data = request.get_json()
        SMOKE = data['SMOKE']
        CALC = data['CALC']
        NCP = data['NCP']
        CH2O = data['CH2O']
        FAF = data['FAF']
        TUE = data['TUE']
        MTRANS = data['MTRANS']
        FCVC = data['FCVC']

        #data_dict = {'SMOKE': SMOKE, 'CALC': CALC, 'NCP': NCP, 'CH2O': CH2O, 'FAF': FAF, 'TUE': TUE, 'MTRANS': MTRANS, 'FCVC': FCVC}
        data_dict = {'SMOKE': [SMOKE], 'CALC': [CALC], 'NCP': [NCP], 'CH2O': [CH2O], 'FAF': [FAF], 'TUE': [TUE], 'MTRANS': [MTRANS], 'FCVC': [FCVC]}


        df = pd.DataFrame(data=data_dict)
        
        df_processed = preprocesar_datos(df)
        
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
