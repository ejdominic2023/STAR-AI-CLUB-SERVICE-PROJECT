import nltk
from nltk.stem import WordNetLemmatizer
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize the chatbot
def initialize_chatbot():
    # Perform any initialization tasks, such as loading data, models, or training
    nltk.download('punkt')
    nltk.download('wordnet')

def is_word(variable):
    if not variable.isalpha():
        return False
    else:
        return True

# Preprocess and tokenize user input
def preprocess_input(input_text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(input_text.lower())
    preprocessed_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return preprocessed_tokens

# Generate bot response
def generate_response(user_input):
    # Define the intents
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
            "examples": ["Tell me a story about", "Can you tell me a story about"],
            "responses": ["Sure, I can tell you a story about anything! One moment..."],
        },
        "compliments": {
            "examples": ["That story was awesome", "I loved the story"],
            "responses": ["Thanks! I appreciate your encouraging words. What was your favorite part?", "Awesome! Glad to hear it! What was your favorite part?"],
        },
        "feedback": {
            "examples": ["It was", "My favorite part was", "It is", "My favorite part is"],
            "responses": ["Cool!", "Nice!"]
        },
        "complaints": {
            "examples": ["I am unhappy with the story", "This story is terrible"],
            "responses": ["I'm sorry to hear that. Please tell me what you did not like about it, and I will tell you an even better story next time!", "Thanks for letting me know! Please tell me what you did not like about it, and I will tell you an even better story next time!"],
        }
    }

    # Check user input against intents
    for intent, data in intents.items():
        for example in data["examples"]:
            if example.lower() in user_input.lower():
                return random.choice(data["responses"])

    # If no intent is detected, provide a default response
    return "I'm sorry, I didn't understand that. Can you please rephrase or ask something else?"

# Start the chatbot
def run_chatbot():
    print("Chatbot: Hello! How can I assist you today?")

    while True:
        user_input = input("User: ")
        #print("user_input.lower(): " + user_input.lower())

        if user_input.lower() == 'bye':
            print("Chatbot: Goodbye!")
            break

        #preprocessed_input = preprocess_input(user_input)
        bot_response = generate_response(user_input)

        print("Chatbot: " + bot_response)

# Main function
if __name__ == '__main__':
    initialize_chatbot()
    run_chatbot()
