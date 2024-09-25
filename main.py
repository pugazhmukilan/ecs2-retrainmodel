
from fastapi import FastAPI, Response, status, HTTPException, Depends
from fastapi.params import Body
from pydantic import BaseModel
from typing import List, Optional
from random import randrange
import pandas as pd
import traning

import modelbit
class PredictionInput(BaseModel):
    values: List[float] 
# Define data models
class Feedback(BaseModel):
    moisture: int
    temperature: int
    humidity: int
    action: str

class RetrainData(BaseModel):
    feedbacks: List[Feedback]
def convert_retrain_data_to_dataframe(retrain_data: RetrainData):
    # Extract list of feedbacks
    feedbacks = retrain_data.feedbacks

    # Convert list of Feedback models to a list of dictionaries
    feedbacks_dicts = [feedback.dict() for feedback in feedbacks]

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(feedbacks_dicts)
    
    return df
def convert_to_retrain_data(data_2d: List[List]) :
    # Step 2: Convert the 2D array into Feedback objects
    feedback_list = [Feedback( moisture=row[0],temperature=row[1], humidity=row[2], action=row[3]) for row in data_2d]
    
    # Step 3: Create RetrainData object from the feedback list and return it
    retrain_data = RetrainData(feedbacks=feedback_list)
    df = convert_retrain_data_to_dataframe(retrain_data=retrain_data)
    return df


app = FastAPI()


@app.get("/")
def root():
    return {"message":"hello world"}



@app.post("/retrainmodel",status_code=status.HTTP_201_CREATED)
def retrain(data:RetrainData):

    data = convert_retrain_data_to_dataframe(data)
    df = pd.read_csv('C:/Users/Pugazh Mukilan/Desktop/SEM 5/ML/Research Paper/app/datasetf.csv')
    
    combined_df = pd.concat([data,df],axis = 0, ignore_index=True)
    
    
    finalmodel = traning.retraing_model(combined_df)
    #deploy it in the mdoelbit
    # print(traning.predict([54,22,70]))
    
    
    
    return{"message": "successfully retrained"}

@app.get("/predict",status_code=status.HTTP_201_CREATED)
def retrain(data:List[int]):

    
    prediction = traning.predict(data)
    return {"prediction": prediction[0]}