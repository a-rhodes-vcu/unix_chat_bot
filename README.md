# Unix Chatbot
This bot provides simple unix commanands when prompted by the user. The bot is capable of providing commands to list files, view contents of files and replace words in a file. It can also help with commands to copy, move and delete a file. 


## Motivation
I have always been fascinated in Chatbots and have been wanting to create my own. I came across a video of Open AI using their API to translate natural language to unix commands and I wanted to try to build a more simplified version of this using Python's data science took kit.
 
## Image
![image_1](https://github.com/a-rhodes-vcu/unix_chat_bot/blob/main/images/ScreenShot.png)

## Code walkthrough
The intents.json file contains key/value pairs of what the user could say to the chatbot (patterns), what the chatbot would say back to the user (responses), a word that groups the intent of the user (tag) and the context of the interaction. 
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
In bashBot.py, the first step is to initialize objects being used

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
First iterate through the outer key of the json file, 'intents' then the inner key, 'patterns'. The patterns are then tokenized or split apart and converted into a list of words. The list is then added to the word list. Then a tuple containing the tokenized pattern along with it's tag is appended to the documents list and the tag is appended to the classes' list. Now that we have the tokenized list of words, I can convert them all to lower case, remove any words that are in the ignore list and find the lemma of each word. Last but not least, create a set of each list (remove duplicates) and sort each list. Finially, it's time to dump the lists into pickle files.

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

## Tech used
Python 3.7

<b>Built with</b>
- [Python](https://www.python.org/)


