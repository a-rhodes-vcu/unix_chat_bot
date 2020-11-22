import nltk
import ssl
import json
import pickle
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


class BuildBotData:
    """ BuildBotData converts the natural language data stored in intents.json and
         turns it into a training set. A model is created using Keras sequential."""

    def __init__(self):

        self.words = []
        self.classes = []
        self.documents = []
        self.training = []
        self.lemmatizer = WordNetLemmatizer()

    def process_data(self):

        """Process the data and store it in a pickle file."""

        # punkt detects work boundries
        # wordnet detects lemmas, a unit of meaning
        nltk.download('punkt')
        nltk.download('wordnet')
        ignore_words = ['?', '!']
        data_file = open('intents.json').read()
        intents = json.loads(data_file)

        for intent in intents['intents']:
            for pattern in intent['patterns']:

                # divide text into a list of sentences
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)

                # append a tuple containing list of sentences with the tag
                self.documents.append((w, intent['tag']))

                # adding tags to the class list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # if not in the ignore list, lemmatize words from the tokenized sentences
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_words]

        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        pickle.dump(self.words, open('words.pkl', 'wb'))
        pickle.dump(self.classes, open('classes.pkl', 'wb'))

    def build_model(self):

        """ Use the lists created in process_data to create a nested list that contains
          the bag of words and output row. """


        self.training = []
        output_empty = [0] * len(self.classes)
        for doc in self.documents:

            # bag of words stores features
            bag_of_words = []

            # extract list of tokenized words
            pattern_words = doc[0]

            # lemmatize each word - create base word, in attempt to represent related words
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words]

            # append to bag of words array with 1, if word match found in current pattern
            for w in self.words:
                bag_of_words.append(1) if w in pattern_words else bag_of_words.append(0)

            # output is a '0' for each tag and '1' for current tag (for each pattern)
            # if the tag in the doc list,is present in the index of the classes list, assign a 1
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1

            self.training.append([bag_of_words, output_row])

        # shuffle our features and turn into np.array
        random.shuffle(self.training)
        # create a grid of values where the first grid is the bag of words
        self.training = np.array(self.training)

    def create_model(self):
        # create train and test lists. X - patterns, Y - intents
        # bag of words
        train_x = list(self.training[:, 0])
        # output row
        train_y = list(self.training[:, 1])

        print("Training data created")

        # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
        # equal to number of intents to predict output intent with softmax
        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_y[0]), activation='softmax'))

        # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # fitting and saving the model
        hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
        model.save('chatbot_model.h5', hist)

        print("model created")


training_data = BuildBotData()
training_data.process_data()
training_data.build_model()
training_data.create_model()
