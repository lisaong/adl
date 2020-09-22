from flask import redirect, render_template
from demo import app

from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired


class ChatInputForm(FlaskForm):
    name = StringField('Reply', validators=[DataRequired()])


@app.route('/', methods=('GET', 'POST'))
def default():
    form = ChatInputForm()
    if form.validate_on_submit():
        return redirect('/')
    return render_template("default.html", form=form)
