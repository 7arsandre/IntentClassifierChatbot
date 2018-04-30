from flask import Flask, render_template, request, jsonify, make_response
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import keras.preprocessing.text as kpt
from keras.models import load_model
from functools import wraps
import tensorflow as tf
import numpy as np
import datetime
import jwt
import json
import sys
import os

#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
#imports load.py
from load import *

#global vars from load.py for easy reuseability
global model, graph, dictionary,bot_answer_array

#initialize these variables
graph, model = init()
dictionary = get_dictionary()
bot_answer_array = get_bot_answers()

#CONSTANTS
MAX_SEQUENCE_LEN = 15

def text_to_index(text):
    words = kpt.text_to_word_sequence(text)
    wordSequence = []
    for word in words:
        if word in dictionary:
            wordSequence.append(dictionary[word])
        else:
            print("'%s' not in training corpus; ignoring." %(word))
    return wordSequence

app = Flask(__name__)
app.config['SECRET_KEY'] = 'megaSecretKey'

#Checks if api_key is present
def api_key_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.args.get('api_key')

        if not api_key:
            return jsonify({'message: ': 'API KEY missing'}), 403

        try:
            data = jwt.decode(api_key, app.config['SECRET_KEY'])
        except:
            return jsonify({'message:' : "API KEY is invalid"}), 403

        return f(*args, **kwargs)
    return decorated

#get_api_key function
@app.route('/get_key')
def login():
    token = jwt.encode({'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=15)},app.config['SECRET_KEY'] )
    return jsonify({'api_key' : token.decode('UTF-8')})

#returns index.html only if valid api_key is present
@app.route('/')
@api_key_required
def index():
    return render_template('index.html')

@app.route('/predict/',methods=['GET','POST'])
def predict():
    userInput =request.json['data']

    # format your input for the neural net
    userInput = text_to_index(userInput)
    #pads input to fit the trained model
    data = pad_sequences([userInput], maxlen=MAX_SEQUENCE_LEN)

    with graph.as_default():
        pred = model.predict(data)
        return (bot_answer_array[np.argmax(pred)][0])

if __name__ == '__main__':
    #define port
    port = int(os.environ.get('PORT', 5000))
	#run the app locally on the given port
    app.run(host='0.0.0.0', port=port)
