# Unix Chatbot
This bot provides simple unix commands when prompted by the user. The bot is capable of providing commands to list files, view contents of files and replace words in a file. It can also help with commands to copy, move and delete a file. 


## Motivation
I have always been fascinated by Chatbots and have been wanting to create my own. I came across a video of Open AI using their API to translate natural language to unix commands([english_to_bash](https://cdn.openai.com/API/English_Bash_Python.mp4)) and I wanted to try to build a more simplified version of this using Python's data science took kit.
 
## Finished Product
Hosted on heroku: https://unix-chat-bot.herokuapp.com
![image_1](https://github.com/a-rhodes-vcu/unix_chat_bot/blob/main/images/ScreenShot.png)

## Code walkthrough
Programs used:
<br>
[flaskChatBot.py](https://github.com/a-rhodes-vcu/unix_chat_bot/blob/main/flaskChatBot.py)
<br>
[intents.json](https://github.com/a-rhodes-vcu/unix_chat_bot/blob/main/intents.json)
<br>
[bashBot.py](https://github.com/a-rhodes-vcu/unix_chat_bot/blob/main/bashBot.py)
<br>

The [intents.json](https://github.com/a-rhodes-vcu/unix_chat_bot/blob/main/intents.json) file contains key/value pairs of what the user could say to the chatbot (patterns), what the chatbot would say back to the user (responses), a word that groups the intent of the user (tag) and the context of the interaction. 
```
{"intents": [
        {"tag": "greeting",
         "patterns": ["Hi there", "How are you", "Is anyone there?","Hey","Hola", "Hello", "Good day"],
         "responses": ["Hello", "Good to meet you", "Hi there, how can I help?"],
         "context": ["intro"]
      
```
In [bashBot.py](https://github.com/a-rhodes-vcu/unix_chat_bot/blob/main/bashBot.py),two nltk coprus' are downloaded. The first one being 'punkt' which divdes text into a list of sentences and the second one is wordnet, which can detect lemmas - a base unit of a word/unit of meaning. The next step is to create a list of things we would like to ignore and then open and read the intents.json file.

```
      def process_data(self):

        """Process the data and store it in a pickle file."""

        # punkt detects sentence boundaries
        # wordnet detects lemmas, a unit of meaning
        nltk.download('punkt')
        nltk.download('wordnet')
        stop = ['?', '!']
        data_file = open('intents.json').read()
        intents = json.loads(data_file)
```
Now it's time to break down the intents.json file, prepare the data and create pickle files. 
First iterate through the outer key and inner key of the json file: 'intents' and 'patterns'. The patterns are then tokenized or split apart and converted into a list of words. The list is then extended to the word list. Then a tuple containing the tokenized pattern along with it's tag is appended to the documents list and the tag is appended to the classes' list. Now that we have the tokenized list of words, I can convert them all to lower case, skip any words that are in the stop list and find the lemma of each word. Last but not least, create a set of each list (remove duplicates) and sort each list. Finally, it's time to dump the lists into pickle files.
```
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

        # store in lists so the model can be created
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))

        # create pickle files so intents.json doesn't have to be processed every time input comes form the user
        pickle.dump(self.words, open('words.pkl', 'wb'))
        pickle.dump(self.classes, open('classes.pkl', 'wb'))
```
build_model creates the inputs for the neural network. I iterate through documents which contains a tuple of tuples. First item in the tuple is the tokenized pattern words and second item is the tag. The lemma is then found for each pattern word and turned into lower case. 
```
   def build_model(self):

        """ Use the lists created in process_data to create a nested list that contains
          the bag of words and output row. """

        self.training = []
        for doc in self.documents:

            # extract items from tuple
            pattern_words, tag = doc[0], doc[1]

            # lemmatize and turn each word to lower case
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words]
```
Next, the bag of words can be made by iterating through the words list using an inner for loop. If the word in the word list is in the current pattern words list from the outer loop append a 1 to the bag of words, else append 0.
```
             # bag of words stores features from the patterns
            bag_of_words = []
            # append to bag of words array with 1, if word match found in current pattern
            for w in self.words:
                bag_of_words.append(1) if w in pattern_words else bag_of_words.append(0)
```
For every iteration output will start as a list of nothing but zeros that is the same size as the classes list, when the current tag in the loop matches the tag found in the classes list, replace the 0 with a 1 at that index. The output row and bag of words is appended to the training list. The training list is then shuffled and turned into a numpy array.
```
            # in every iteration of the for loop create list of empty zeros same size of classes list
            output = [0] * len(self.classes)
            # for the current tag in the loop change 0 to 1
            output[self.classes.index(tag)] = 1

            # append bag of words and output row to training list
            self.training.append([bag_of_words, output])

        # shuffle our features and turn into np.array
        random.shuffle(self.training)
        # create a grid of values where the first grid is the bag of words
        self.training = np.array(self.training)
```
Finally to the neural network! The neural network has three layers, first layer contains 128 neurons, second layer has 64 neurons and last layer contains the same amount of neurons as the length of train_y. The neural network is trained, compiled and then fitted. The model is saved to a .h5 file which is used by the flask app.
```
    def create_model(self):
    
        """Use bag of words list and output list from build_model to create model, train, compile and fit. 
        model is saved in a .h5 file. """
        
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
```
Last but not least, the [flask app](https://github.com/a-rhodes-vcu/unix_chat_bot/blob/main/flaskChatBot.py)!
msg is the user input, which is tokenized and then turned into a bag of words. The bag of words is fed into the model and a list of probabilities is returned. The highest probability for the tag is then used to return the chat bot response.
```
@app.route("/get")
def get_bot_response():

    """receive messages from the user and return messages from the bot"""

    msg = request.args.get("msg")  # get data from input
    res = chatbot_response(msg)
    return str(res)
```

## Improvements
1. The json file was limited to just a few possible unix commands that could be returned to the user, so adding more would greatly improve the versatility of the chatbot.
2. The sequential neural network implenmented for this is pretty simple. It would be interesting to see how a recurrent neural network impacts the user experience.
3. The HTML/CSS leaves a lot to be desired. Adding more styling to make the chatbot more user friendly would be an added bonus.

## Tech used
[Python](https://www.python.org/) 3.7


<h5>If you made it to the end, thanks for reading! Feel free to reach out for questions or suggesstions! This was such a fun project to work on and I look forward to more future NLP/neural network projects.</h5>
