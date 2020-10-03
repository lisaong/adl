from flask import Flask
import os
import secrets

# prepare the upload folder
upload_folder = os.path.join(os.path.curdir, 'uploads')
os.makedirs(upload_folder, exist_ok=True)

app = Flask(__name__)

# for uploads
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB

# for sessions
app.config['SECRET_KEY'] = secrets.token_urlsafe(16)

from demo import views
