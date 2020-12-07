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
intents = json.loads(open('intents.json').read())
import pickle
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.05
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])

            #break
            return result


def chatbot_response(msg):
    
    """msg is the input from the user. that is passed into predict class, which predicts the intent of the request
    ints is then passed into getResponse which is the reponse from the chat bot"""
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

@app.route("/")
def index():
    
    """render the html and css"""
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    
    """ this function takes user input and then returns the bot response """
    msg = request.args.get("msg")  # get data from input
    res = chatbot_response(msg)
    print(res)
    return str(res)

if __name__ == '__main__':
    app.run(debug = True)

