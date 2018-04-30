# Imports
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Activation, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.models import model_from_json
from keras.utils import to_categorical
import numpy as np
import json
import os

kindly_bot = "src/kindly_bot.json"

with open(kindly_bot) as json_file:
    data = json.load(json_file)

user_input_array = []
bot_answer_array = []
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
k = 0
for i in data["dialogues"]:
    if("nb" in data["dialogues"][k]["samples"]):
        label_id = len(labels_index)
        labels_index[data["dialogues"][k]["title"]] = label_id
        for samples in data["dialogues"][k]["samples"]["nb"]:
            user_input_array.append(samples)
            bot_answer_array.append(data["dialogues"][k]["replies"]["nb"])
            labels.append(label_id)
    k += 1

#Sentences are padded to 15 words
MAX_SEQUENCE_LEN = 15
#Only the 4k top words in the input file are vectorized
MAX_NUM_WORDS = 4000

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(user_input_array)

word_index = tokenizer.word_index

#Dumps a json dictionary with the words
with open('model/dictionary.json', 'w', encoding = 'utf8') as dictionary_file:
    json.dump(word_index, dictionary_file, ensure_ascii=False)

#Converts each word into a sequence, done for the network
def convert_text_to_index_array(text):
    return [word_index[word] for word in text_to_word_sequence(text)]

allSequences = []
for text in user_input_array:
    wordSequence = convert_text_to_index_array(text)
    allSequences.append(wordSequence)

allSequences = np.asarray(allSequences)
#Pads the sequence of words
data = pad_sequences(allSequences, maxlen=MAX_SEQUENCE_LEN)
#The bot answers are turned into one-hot encoded vectors
labels = to_categorical(labels, num_classes = len(labels_index))

#Time to shuffle, so that dataset isn' predictable for the network
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
VALIDATION_SPLIT = 0.2
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

#Should have used a exclusive test set, but decided to use as much as possible data in the training set... So val_set == test_set, not optimal for test, I knows
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


print(x_train.shape)
print(x_train.shape[0])

#x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
#x_val = np.reshape(x_val, (x_val.shape[0], 1, x_val.shape[1]))

print(x_train.shape)
print(x_train.shape[0])
print(len(labels_index))


print(str(x_train[0]))
print('Build model...')
model = Sequential()
model = Sequential()
model.add(Embedding(MAX_NUM_WORDS, 512))
model.add(LSTM(512, return_sequences = True))    # returns a sequence of vectors of dimension 512
model.add(Dropout(0.2))
model.add(LSTM(1024, return_sequences = True))   # returns a sequence of vectors of dimension 512
model.add(Dropout(0.2))
model.add(LSTM(512))                             # returns a single vector of dimension 32
model.add(Dense(324, activation='softmax'))      #output num_classes
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

print('Train...')
batch_size = 24
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=20,
          validation_data=(x_val, y_val),
          shuffle = True)

score, acc = model.evaluate(x_val, y_val,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# serialize model to JSON
model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/model.h5")
print("Saved model to disk")
