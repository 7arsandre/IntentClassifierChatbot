# Intent-classifier for the [Convertelligence](https://www.convertelligence.no/) chatbot

## Making of a model

Made possible with the power of [Keras](https://keras.io/). The network this model
is based on, is a so called `long short-term memory` (LSTM), which is a
recurrent neural network. Main purpose of the LSTM network is to making the
cells "remember" values over arbitrary times, hence the word memory in LSTM.

### Data pre-processing

Data was generated into two different arrays, `user input` and `bot answer`.
The `user input` was vectorized with consideration of the words frequency in the list.
The `bot answer` was one-hot encoded so that it was easy to train on.
I was made aware of that I had to have a training set, validation set and a
test set. But because of limited amount of data, I decided to only use
a training- and validation set.

### The model
To prevent overfitting I decided to implement a dropout between each layer.
Furthermore, I went for a structure in the network that began and ended with
the same number of neurons, and had a higher number of neurons in the middle.

For the activation function I thought `softmax` was a suitable choice, since
we're a classification problem.

A note on accuracy; During training it had an accuracy above 90% percent,
and its final loss was about 0.09, as we can see from the [output](./output_train.txt).
In practice, the accuracy doesn't seem to be that good. This is probably a
combination between a bit poor network and a relative small dataset.

## Creating an API
Made possible with the power of [Flask-RESTful](https://flask-restful.readthedocs.io/en/latest/).

### API key
The `API KEY` was generated with the power of [JSON web tokens](https://jwt.io/).
The key I implemented lasts for 15 minutes, and you have to generate a new
one after that.

### User interface
Because of limited time, I decided to make a truly simple and non-aestetich UI.
The user interface (UI) is just an user input box, a send button and a
bot-respons field.

## Deploy to [Google Cloud](https://cloud.google.com/)

After you have downloaded the [Google Cloud](https://cloud.google.com/sdk/docs/) software development kit (SDK), it's two command line arguments that would deploy
the current repository you're located in to the Google Cloud.

`$ gcloud app deploy` to deploy our app to the App Engine. This command
builds a container image , and then deploys it to the App Engine environment.

Then, `$ gcloud app browse` makes it possible for you to launch and view your
app in the web browser at `https://YOUR_PROJECT_ID.appspot.com`


## How to run
NB! Since the dataset I used for this model is not uploaded to this repo.,
the following steps won't work unless you modify the code to adapt to your
own dataset.   

To start the API, run `$ python app.py`. Once the API is launced you have
to get a valid api key to gain access to the API. Use the terminal, or
your web-browser and do the following:
<pre>
***if terminal***
$ curl 0.0.0.0:5000/get_key

***if web-bowser write in url***
https://0.0.0.0:5000/get_key
</pre>

An JSON-output like this should be shown:
<pre>
{
  "api_key": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE1MjQ5MjUzNDh9.w8OpPIZWg5GxofhA6smhvFPg3J7xEZSxhq3CoAn-0jk"
}

NB! This is just an example API KEY that will not be valid at the time you read this
</pre>

Fetch that key and copy-past into the url bar like

<pre>
https://0.0.0.0:5000/?api_key=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE1MjQ5MjUzNDh9.w8OpPIZWg5GxofhA6smhvFPg3J7xEZSxhq3CoAn-0jk
</pre>

The API should now be available for you to use in 15 minutes before you have to generate a new key.
