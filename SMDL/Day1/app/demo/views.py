from flask import redirect, render_template
from demo import app
from demo import ml

from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired


class ChatInputForm(FlaskForm):
    reply = StringField('Reply', validators=[DataRequired()])


model = ml.TFModel()


@app.route('/', methods=('GET', 'POST'))
def default():
    form = ChatInputForm()
    if form.validate_on_submit():
        print(form.reply.data)
        result = model.predict(form.reply.data)
        print(result)
        return redirect('/')
    return render_template("default.html", form=form)
