from flask import render_template, request, json
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

    # use a random word from the reply text
    tokens = text.split()
    seed_word = random.sample(tokens, k=1)

    # call the TFModel class to predict
    predictions = model.predict(seed_word)
    print(predictions)
    return json.dumps({'predictions': predictions})
