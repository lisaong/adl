from flask import redirect, render_template, request, json
from demo import app
from demo import ml


model = ml.TFModel()


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
