from flask import redirect, render_template, request, json
from demo import app
from demo import ml
import os
import random


model_dir = os.path.join(os.getcwd(), 'demo', 'model')
model = ml.TFModel(model_dir=model_dir)


@app.route('/', methods=['GET'])
def default():
    return render_template("default.html")


@app.route('/reply', methods=['POST'])
def reply_chat():
    text = request.form['reply']

    # call the TFModel class to predict
    predictions = model.predict(text)
    print(predictions)
    return json.dumps({'predictions': predictions})
