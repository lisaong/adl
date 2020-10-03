from flask import Flask
import os

# prepare the upload folder
upload_folder = os.path.join(os.path.curdir, 'uploads')
os.makedirs(upload_folder, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB

from demo import views
