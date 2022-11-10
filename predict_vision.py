import roboflow
import json
from decouple import config

rf_1 = roboflow.Roboflow(api_key=config('api_key_1'))
project_1 = rf_1.workspace().project("violence-not_violence-ziv7b")
model_1 = project_1.version(2).model

rf_2 = roboflow.Roboflow(api_key=config('api_key_2'))
project_2 = rf_2.workspace().project("violence-not_violence")
model_2 = project_2.version(3).model


def check_image_toxic_url(url):
    flag=0

    # infer on an image hosted elsewhere
    probs_1 = model_1.predict(url, hosted=True)
    probs_1 = json.loads(str(probs_1))

    # infer on an image hosted elsewhere
    probs_2 = model_2.predict(url, hosted=True)
    probs_2 = json.loads(str(probs_2))
    
    if probs_2['predictions']['unsafe']['confidence'] >= 0.75 or probs_1['predictions']['violence']['confidence'] >= 0.75:
        flag=1

    return flag


def check_image_toxic_file(filepath):
    flag=0
    # infer on an image hosted elsewhere
    probs_1 = model_1.predict(filepath)
    probs_1 = json.loads(str(probs_1))

    # infer on an image hosted elsewhere
    probs_2 = model_2.predict(filepath)
    probs_2 = json.loads(str(probs_2))
    
    if probs_2['predictions']['unsafe']['confidence'] >= 0.75 or probs_1['predictions']['violence']['confidence'] >= 0.75:
        flag=1
    return flag