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

        # punkt detects sentence boundies
        # wordnet detects lemmas, a unit of meaning
        nltk.download('punkt')
        nltk.download('wordnet')
        stop = ['?', '!']
        data_file = open('intents.json').read()
        intents = json.loads(data_file)

        for intent in intents['intents']:
            for pattern in intent['patterns']:

                # divide text into a list of sentences
                w = nltk.word_tokenize(pattern)
                self.words.extend(w)

                # append a tuple containing list of sentences with the tag
                self.documents.append((w, intent['tag']))

                # add tag if not already in classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # if not in the ignore list, lemmatize words from the tokenized sentences and turn everything to lower case
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in stop]

        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        pickle.dump(self.words, open('words.pkl', 'wb'))
        pickle.dump(self.classes, open('classes.pkl', 'wb'))

    def build_model(self):

        """ Use the lists created in process_data to create a nested list that contains
          the bag of words and output row. """

        self.training = []
        for doc in self.documents:

            # extract items from tuple
            pattern_words, tag = doc[0], doc[1]

            # lemmatize and turn each word to lower case
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words]

            # bag of words stores features from the patterns
            bag_of_words = []
            # append to bag of words array with 1, if word match found in current pattern
            for w in self.words:
                bag_of_words.append(1) if w in pattern_words else bag_of_words.append(0)

            # in every iteration of for loop create list of empty zeros same size of classes list
            output = [0] * len(self.classes)
            # for the current tag in the loop change 0 to 1
            output[self.classes.index(tag)] = 1

            # append bag of words and output row to training list
            self.training.append([bag_of_words, output])

        # shuffle our features and turn into np.array
        random.shuffle(self.training)
        # create a grid of values where the first grid is the bag of words
        self.training = np.array(self.training)

    def create_model(self):
        # gather lists from np.array to train and test
        # bag of words
        train_x = list(self.training[:, 0])
        # output row
        train_y = list(self.training[:, 1])

        print("Training data created")

        # Create model - 3 layers
        # used the sequential model which is a stack of layers feeding linearly from one to the next
        # the output of one layer is input to next layer
        model = Sequential()
        # first layer 128 neurons
        # input shape is length of the bag of words
        # relu is the default activation function
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        # drop out 50% of neurons to prevent over fitting and can lower variability of neural network
        model.add(Dropout(0.5))
        # second layer has 64 neurons
        model.add(Dense(64, activation='relu'))
        # third output layer has the number of neurons equal to the number of tags
        # softmax is the default activation function for the third later
        model.add(Dense(len(train_y[0]), activation='softmax'))

        # Compile model using SGD optimizer and cross-entropy.
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # fit and save the model
        hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=2)
        model.save('chatbot_model.h5', hist)

        print("model created")


training_data = BuildBotData()
training_data.process_data()
training_data.build_model()
training_data.create_model()
