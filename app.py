from flask import Flask, jsonify
from flask import render_template
from flask import request
import pickle as pk

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    newsText = request.args.get('inputText')
    model = pk.load(open("model.p", 'rb'))
    predictedClass = model.predict(newsText)
    return render_template('index.html', predictedClass=predictedClass)

