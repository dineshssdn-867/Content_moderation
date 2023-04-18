import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


model_mobilenetv3large = tf.keras.models.load_model('model_mobilenetv3large.h5')

def check_image_toxic_url(url):
    random_string_for_file_name = 'test_image'
    img_path = get_file(random_string_for_file_name,url)
    img = image.load_img(img_path,target_size=(448,448))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    images = np.vstack([img])
    preds = model_mobilenetv3large.predict(images, batch_size=1)
    classes =  np.argmax(preds, axis=-1)
    if abs(preds[0][0]-preds[0][1]) > 0.989:
        os.remove(img_path)
        return 0
    elif classes[0] == 0:
        os.remove(img_path)
        return 1
    else:
        os.remove(img_path)
        return 0


def check_image_toxic_file(path):
    img = image.load_img(path,target_size=(448,448))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    preds = model_mobilenetv3large.predict(img)
    classes =  np.argmax(preds, axis=-1)
    print(preds)
    if abs(preds[0][0]-preds[0][1]) > 0.989:
        return 0
    elif classes[0] == 0:
        return 1
    else:
        return 1
