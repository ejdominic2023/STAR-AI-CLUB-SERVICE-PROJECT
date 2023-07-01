import random

# Story templates
templates = [
    "Once upon a time, {character} embarked on an adventure {location}. {character} was {emotion} and encountered {obstacle}. In the end, {character} learned {lesson}.",
    "In a {location} far away, {character} found themselves in a peculiar situation. Feeling {emotion}, {character} had to overcome {obstacle} and discover {lesson}.",
    "Long ago, in the land of {location}, there lived {character}. One day, {character} felt {emotion} and faced a daunting {obstacle}. Through this journey, {character} realized {lesson}."
]

# Character names, emotions, obstacles, and lessons
characters = ["Alice", "Bob", "Eva", "Max"]
emotions = ["happy", "sad", "excited", "scared"]
obstacles = ["a fierce dragon", "a mysterious maze", "a treacherous storm", "an enchanted curse"]
lessons = ["the value of friendship", "the importance of perseverance", "the power of forgiveness",
           "the beauty of diversity"]


def generate_story(character, location):
    # Select a random template
    template = random.choice(templates)

    # Generate random elements
    emotion = random.choice(emotions)
    obstacle = random.choice(obstacles)
    lesson = random.choice(lessons)

    # Fill in the template with user input and random elements
    story = template.format(character=character, location=location, emotion=emotion, obstacle=obstacle, lesson=lesson)
    return story


# Main program loop
while True:
    user_character = input("Enter a character name: ")
    user_location = input("Enter a location: ")

    # Generate and display the story
    story = generate_story(user_character, user_location)
    print("\nOnce upon a time...\n")
    print(story)

    # Ask if the user wants to generate another story
    another_story = input("\nDo you want to generate another story? (yes/no): ")
    if another_story.lower() != "yes":
        break
