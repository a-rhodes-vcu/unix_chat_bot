# Unix Chatbot
This bot provides simple unix commands when prompted by the user. The bot is capable of providing commands to list files, view contents of files and replace words in a file. It can also help with commands to copy, move and delete a file. 


## Motivation
I have always been fascinated by Chatbots and have been wanting to create my own. I came across a video of Open AI using their API to translate natural language to unix commands and I wanted to try to build a more simplified version of this using Python's data science took kit.
 
## Image
![image_1](https://github.com/a-rhodes-vcu/unix_chat_bot/blob/main/images/ScreenShot.png)

## Code walkthrough
The [intents.json](https://github.com/a-rhodes-vcu/unix_chat_bot/blob/main/intents.json) file contains key/value pairs of what the user could say to the chatbot (patterns), what the chatbot would say back to the user (responses), a word that groups the intent of the user (tag) and the context of the interaction. 
```
{"intents": [
        {"tag": "greeting",
         "patterns": ["Hi there", "How are you", "Is anyone there?","Hey","Hola", "Hello", "Good day"],
         "responses": ["Hello", "Good to meet you", "Hi there, how can I help?"],
         "context": ["intro"]
        },
        {"tag": "goodbye",
         "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"],
         "responses": ["See you!", "Have a nice day", "Bye!"],
         "context": ["close"]
        },


```
In [bashBot.py](https://github.com/a-rhodes-vcu/unix_chat_bot/blob/main/bashBot.py), the first step is to initialize the attributes of the BuildBotData class

```
class BuildBotData:
    
    """ BuildBotData converts the natural language data stored in intents.json and
         turns it into a training set. A model is created using Keras sequential."""

    def __init__(self):

        self.words = []
        self.classes = []
        self.documents = []
        self.training = []
        self.lemmatizer = WordNetLemmatizer()

```
Starting on line 38 of bashBot.py two nltk coprus' are downloaded. The first one being 'punkt' which divdes text into a list of sentences and the second one is wordnet, which can detect lemmas - a base unit of a word/unit of meaning. The next step is to create a list of things we would like to ignore and then open and read the intents.json file.

```
       def process_data(self):

        """Process the data and store it in a pickle file."""

        # punkt detects sentence boundies
        # wordnet detects lemmas, a unit of meaning
        nltk.download('punkt')
        nltk.download('wordnet')
        ignore_words = ['?', '!']
        data_file = open('intents.json').read()
        intents = json.loads(data_file)
```
Now it's time to break down the intents.json file, prepare the data and create pickle files. 
First iterate through the outer key and inner key of the json file: 'intents' and 'patterns'. The patterns are then tokenized or split apart and converted into a list of words. The list is then extended to the word list. Then a tuple containing the tokenized pattern along with it's tag is appended to the documents list and the tag is appended to the classes' list. Now that we have the tokenized list of words, I can convert them all to lower case, skip any words that are in the ignore list and find the lemma of each word. Last but not least, create a set of each list (remove duplicates) and sort each list. Finially, it's time to dump the lists into pickle files.

```
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
```
Now it's time to create the inputs for the neural network. First thing to so is create an empty training list. Next thing to do is iterate through documents which contains a tuple of tuples. First item in the tuple is the tokenized pattern words and second item is the tag. The lemma is then found for each pattern word and turned into lower case. 
```
    def build_model(self):

        """ Use the lists created in process_data to create a nested list that contains
          the bag of words and output row. """

        self.training = []
        for doc in self.documents:

            # bag of words stores features from the patterns
            bag_of_words = []

            # extract items from tuple
            pattern_words, tag = doc[0], doc[1]

            # lemmatize each word - create base word, in attempt to represent related words
            pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words
```
Next, the bag of words can be made by iterating through the words list using an inner for loop. If the word in the word list is in the current pattern words list from the outer loop append a 1 to the bag of words, else append 0.
```
            # append to bag of words array with 1, if word match found in current pattern
            for w in self.words:
                bag_of_words.append(1) if w in pattern_words else bag_of_words.append(0)
```
For every iteration output_row will start as a list of nothing but zeros that is the same size as the classes list, when the current tag in the loop matches the tag found in the classes list, replace the 0 with a 1 at that index. The output row and and bag of words is appeneded to the training list. The training list is then shuffled and turned into a numpy array.
```
            # output is a '0' for each tag and '1' for current tag (for each pattern)
            # if the tag in the doc list, is present in the index of the classes list, assign a 1
            output_row = [0] * len(self.classes)
            output_row[self.classes.index(tag)] = 1

            self.training.append([bag_of_words, output_row])

        # shuffle our features and turn into np.array
        random.shuffle(self.training)
        # create a grid of values where the first grid is the bag of words
        self.training = np.array(self.training)

```

## Tech used
Python 3.7

<b>Built with</b>
- [Python](https://www.python.org/)


