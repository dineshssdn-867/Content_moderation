from distutils.log import debug
import os
import random
import string
from flask import Flask, request, jsonify, render_template
from predict_vision import check_image_toxic_url, check_image_toxic_file
from predict_text import check_image_toxic_text
from flask_cors import CORS

content_moderator_app = Flask(__name__)
CORS(content_moderator_app)

@content_moderator_app.route("/")
@content_moderator_app.route("/home")
def home():
    return render_template("index.html")

@content_moderator_app.route('/predict_toxicity_image',methods=['POST'])
def predict_toxicity_image():
    error = ""
    result = "Content is relevant"
    try:
        image_file = request.files['file']
        file_path='file_'+''.join(random.choices(string.ascii_uppercase + string.digits, k=7))+'.jpg'
        image_file.save(file_path)
        probs = check_image_toxic_file(file_path)
        os.remove(file_path)
        if probs > 0.75:
            result = "Content is not relevant. Please remove content"
        return render_template("index.html", result_image=result, error_image=error)
    except Exception as e:
        os.remove(file_path)
        try:
            url = request.form.get('url')
            probs = check_image_toxic_url(url)
            if probs > 0.75:
                result = "Content is not relevant. Please remove content"
            return render_template("index.html", result_image=result, error_image=error)
        except Exception as e:
            error = "Please enter a valid input in the fields"
            return render_template("index.html", result_image=result, error_image=error)
    

@content_moderator_app.route('/predict_toxicity_text',methods=['POST'])
def predict_toxicity_text():
    error = ""
    result = "Content is relevant"
    try:
        text = request.form.get('comment')
        probs = check_image_toxic_text(text)
        if probs > 0.75:
            result = "Content is not relevant. Please remove content"
        return render_template("index.html", result_text=result, error_text=error)
    except:
        error = "Please enter a valid input in the fields"
        return render_template("index.html", result_text=result, error_text=error)


if __name__ == '__main__':
    content_moderator_app.run(debug=True)
    
