from flask import Flask, render_template

# create our Flask class object
app = Flask(__name__)


@app.route('/')
def default():
    return render_template("default.html")


if __name__ == '__main__':
    app.run(debug=True)