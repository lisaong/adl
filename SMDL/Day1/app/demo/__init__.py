from flask import Flask
import os

# for CSRF protection (FlaskForm)
SECRET_KEY = os.urandom(32)

# create our Flask class object
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY

from demo import views
