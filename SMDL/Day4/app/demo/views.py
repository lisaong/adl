from flask import render_template, request, url_for
from demo import app
from demo import ml
import os

# https://flask.palletsprojects.com/en/1.1.x/patterns/fileuploads/
from werkzeug.utils import secure_filename

model_dir = os.path.join(os.getcwd(), 'demo', 'model')
model = ml.TFModel(model_dir=model_dir)


@app.route('/', methods=['GET'])
def default():
    return render_template("default.html")


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filename = secure_filename(file.filename)
    complete_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(complete_filename)

    # call the TFModel class to predict
    prediction, probability = model.predict(complete_filename)
    print(prediction, probability)

    base_url = url_for('default', _external=True)
    filename_url = f'{base_url}uploads/{filename}'
    print(filename_url)

    return render_template('result.html',
                           prediction=f'{prediction} {probability:.3f}',
                           url=filename_url)
