# deploy using fastapi
import pickle
from fastapi import FastAPI, Body
import numpy as np 

loaded_model = pickle.load(open('diabetes.sav', 'rb'))

app = FastAPI()

@app.post('/diabetes/v1/predict')
async def predict(Glucose: int = Body(), BMI: float = Body(), Age: int = Body()):
  #---create the features list for prediction---
  features_list = [Glucose,BMI,Age]
  #---get the prediction class---
  prediction = loaded_model.predict([features_list])
  #---get the prediction probabilities---
  confidence = loaded_model.predict_proba([features_list])

  res = {
    'prediction': int(prediction[0]),
    'confidence': str(round(np.amax(confidence[0]) * 100 ,2))
  }
  return res  