import keras.models
import tensorflow as tf
from keras.models import model_from_json
import json

kindly_bot = "src/kindly_bot.json"

#Loads and compile model. Returns the essentials for predicting
def init():
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # and create a model from that
    loaded_model = model_from_json(loaded_model_json)
    # and weight your nodes with your saved values
    loaded_model.load_weights("model/model.h5")

    loaded_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    graph = tf.get_default_graph()

    return graph, loaded_model

#The dictionary used for training
def get_dictionary():
    with open('model/dictionary.json', 'r') as dictionary_file:
        dictionary = json.load(dictionary_file)
    return dictionary

#The static bot answers used for training gets returned
def get_bot_answers():
    with open(kindly_bot) as json_file:
        data = json.load(json_file)

    user_input_array = []
    bot_answer_array = []
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    k = 0

    for i in data["dialogues"]:
        if("nb" in data["dialogues"][k]["samples"]):
            for samples in data["dialogues"][k]["samples"]["nb"]:
                bot_answer_array.append(data["dialogues"][k]["replies"]["nb"])
        k += 1
    return bot_answer_array
