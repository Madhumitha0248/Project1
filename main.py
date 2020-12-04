
import uvicorn
from fastapi import FastAPI
from Readings import Reading
import numpy as np
import pickle
import pandas as pd

app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier =pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message' : 'Hello, Stranger'}

@app.get('/{name}')
def get_name(name:str):
    return {'welcome' : f'{name}'}

@app.post('/predict')
def predict_risk(data:Reading):
    data = data.dict()
    age = data['age']
    sex = data['sex']
    cp = data['cp'] 
    trestbps = data['trestbps']
    chol = data['chol'] 
    fbs = data['fbs']
    restecg = data['restecg']
    thalach = data['thalach']
    exang = data['exang']
    oldpeak = data['oldpeak']
    slope = data['slope']
    ca = data['ca'] 
    thal = data['thal']
    print(classifier.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]))
    prediction = classifier.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    if(prediction[0]>0.5):
        prediction = "The person has a risk of heart disease"
    else:
        prediction = "The person does not have any risk of heart disease"
    return{
        'prediction': prediction}

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)

