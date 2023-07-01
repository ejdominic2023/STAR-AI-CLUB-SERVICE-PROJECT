# I) Import the necessary packages/dependencies
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import tensorflow
import numpy as np
import json
import PyPDF2
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import random
from sklearn.model_selection import train_test_split

# II) Define the intents
intents = {
    "greeting": {
        "examples": ["Hi", "Hello", "Hey"],
        "responses": ["Hello!", "Hi there! What kind of story would you like for me to tell you today?", "Hey! How can I be of assistance?"],
    },
    "farewell": {
        "examples": ["Goodbye", "Bye", "See you later"],
        "responses": ["Goodbye, have a great day!", "Bye, take care!", "See you later!"],
    },
    "thankyou": {
        "examples": ["Thank you", "Thanks a lot"],
        "responses": ["You're welcome!", "Glad to be of assistance.", "Anytime!"],
    },
    "story_options": {
        "examples": ["Tell me a story about...", "Can you tell me a story about..."],
        "responses": ["Sure, I can tell you a story about anything! One moment..."],
    },
    "compliments": {
        "examples": ["That story was awesome!", "I loved the story!"],
        "responses": ["Thanks! I appreciate your encouraging words. What was your favorite part?", "Awesome! Glad to hear it! What was your favorite part?"],
    },
    "complaints": {
        "examples": ["I am unhappy with the story", "This story is terrible"],
        "responses": ["I'm sorry to hear that. Please tell me what you did not like about it, and I will tell you an even better story next time!", "Thanks for letting me know! Please tell me what you did not like about it, and I will tell you an even better story next time!"],
    }
}

# Convert 'intents' dictionary to numpy array
numpy_intents = np.array(list(intents.items()))

# III) Preprocess the data.
nltk.download('punkt')
nltk.download('stopwords')

# Open the PDF file and read its contents
pdf_file = open('DukeECE.pdf', 'rb')
pdf_reader = PyPDF2.PdfFileReader(pdf_file)
text = ''
for page in range(pdf_reader.numPages):
    page_obj = pdf_reader.getPage(page)
    text += page_obj.extractText()
    print("Text: " + text)

# Tokenize the text into words
words = word_tokenize(text)
print("Words: " + str(words))

# Remove stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.lower() not in punctuations]

# Print the preprocessed text
print(filtered_words)

# 1) Tokenize the input. The nltk.word_tokenize() method will split the input text into individual words.
def tokenize_input(input_text):
    tokens = nltk.word_tokenize(input_text.lower())
    return tokens

# 2) Remove punctuation and stopwords. The string.punctuation constant contains a string of all punctuation characters.
# We use it to remove punctuation from the tokens. We also remove stopwords, which are common words like "a", "the", and
# "and" that are unlikely to convey any meaningful information about the user's intent.
def remove_punctuation_and_stopwords(tokens):
    stopwords_list = set(stopwords.words("english"))
    tokens_without_punctuation = []
    for token in tokens:
        if token not in string.punctuation:
            if token not in stopwords_list:
                tokens_without_punctuation.append(token)
    return tokens_without_punctuation

# 3) Lemmatize the tokens. Lemmatization is the process of reducing words to their base form (or lemma). For example, the
# words "running", "ran", and "runs" all have the same lemma, which is "run". We use the WordNetLemmatizer from the nltk
# library to lemmatize the tokens.
def lemmatize_tokens(tokens_without_punctuation):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for token in tokens_without_punctuation:
        lemmatized_token = lemmatizer.lemmatize(token)
        lemmatized_tokens.append(lemmatized_token)
    return lemmatized_tokens

# 4) Put it all together. This function puts all the previous steps together to preprocess the user's input text. You can
# call this function on the user's input before passing it to your machine learning model to classify the user's intent.
def preprocess_input(input_text):
    tokens = tokenize_input(input_text)
    tokens_without_punctuation = remove_punctuation_and_stopwords(tokens)
    lemmatized_tokens = lemmatize_tokens(tokens_without_punctuation)
    return lemmatized_tokens

# IV) Building a Machine Learning Model

# Step #1: Convert into numpy array.
numpy_filtered_words = np.array(filtered_words)

# Step 2: Vectorize the Text Data
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(numpy_filtered_words)

# Step 3: Prepare the Training Data
# Assuming you have a numpy array called 'train_labels' containing corresponding labels/intents
y = np.array(numpy_intents)

# Determine the minimum number of rows to use for splitting
min_rows = min(X.shape[0], len(y))

# Trim X_train and train_labels to have the same number of rows
X = X[:min_rows]
y = y[:min_rows]

# Step 4: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Choose a Machine Learning Algorithm
clf = MultinomialNB()

# Convert y_train to a 1D array
y_train = np.ravel(y_train)

# Determine the minimum number of rows to use for splitting
min_rows = min(X_train.shape[0], len(y_train))

# Trim X_train and train_labels to have the same number of rows
X_train = X_train[:min_rows]
y_train = y_train[:min_rows]
y_train = [str(label) for label in y_train]

# V) Train the Model
clf.fit(X_train, y_train)

y_test = [str(label) for label in y_test]

# VI) Evaluate the Model
accuracy = clf.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Ste: Use the Model for Inference
# Assuming you have a user input called 'user_input'
#preprocessed_input = preprocess_input(user_input)
#input_vector = vectorizer.transform([preprocessed_input])
#predicted_label = clf.predict(input_vector)
#print("Predicted Label:", predicted_label)

# V) Train the model: Train the model on a dataset of labeled examples that you have created. This dataset should include examples of user input and the corresponding intent.

# VI) Test the model: After training, test the model with some sample inputs to see how it performs. If it doesn't perform well, you may need to retrain the model with more data or adjust its parameters.

# VII) Build a response system: Once you have trained and tested the model, you can build a response system to generate appropriate responses for each intent. You can use a dictionary to map each intent to a set of responses.
# Define a function to process user inputs and generate responses
def generate_response(input_text):
    input_text = input_text.lower()
    for intent, intent_data in intents.items():
        for pattern in intent_data["examples"]:
            if pattern.lower() in input_text:
                return random.choice(intent_data["responses"])
    return "I'm sorry, I don't understand. Could you please rephrase your question?"

# Test the response system with some inputs
print(generate_response("Hello"))
print(generate_response("Thanks for your help."))
print(generate_response("Goodbye!"))
print(generate_response("Can you tell me more about your services?"))