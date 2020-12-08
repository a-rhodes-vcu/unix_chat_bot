from flask import Flask, render_template, request
app = Flask(__name__)

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import nltk
import numpy as np
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import random
import pickle

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('tags.pkl','rb'))
intents = json.loads(open('intents.json').read())


def clean_up_sentence(sentence):

    """tokenize user input to create bag of words"""

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):

    """input from the user has to be converted into a bag of words to feed to model"""

    # tokenize user input
    sentence_words = clean_up_sentence(sentence)
    # create array of zeros of same length of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
    return(np.array(bag))


def predict_class(sentence, model):

    """get the prediction from the model"""

    # get a list of predictions from the model
    res = model.predict(np.array([bow(sentence, words)]))[0]
    print(res)
    error_threshold = 0.05
    # save index and probability if it's is above a certain threshold
    results = [[i, pred] for i, pred in enumerate(res) if pred > error_threshold]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    # go through list of indices and probabilities and save the tag label and probability to a list
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(ints, intents_json):

    """save the tag from ints list. this will be used as the key for the chat bot responses.
    return the chat bot reply"""

    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            return result


def chatbot_response(msg):

    """ ints is a list of tags and probabilities from the model
    res is the chat bot reply"""

    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return res

@app.route("/")
def index():

    """render the html and css"""

    return render_template("index.html")

@app.route("/get")
def get_bot_response():

    """receive messages from the user and return messages from the bot"""

    msg = request.args.get("msg")  # get data from input
    res = chatbot_response(msg)
    return str(res)

if __name__ == '__main__':
    app.run(debug = True)

