from flask import Flask, render_template, redirect
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired

# for CSRF protection (FlaskForm)
import os
SECRET_KEY = os.urandom(32)


class MyForm(FlaskForm):
    name = StringField('name', validators=[DataRequired()])


# create our Flask class object
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY


@app.route('/', methods=('GET', 'POST'))
def default():
    form = MyForm()
    if form.validate_on_submit():
        return redirect('/')
    return render_template("default.html", form=form)


if __name__ == '__main__':
    app.run(debug=True)