import gpt_3.5_turbo as gpt

# OpenAI API Parameters
model_name = "text-davinci-003"
max_tokens = 100  # Adjust the desired length of the generated story

# Function to generate a story based on the specified genre or topic
def generate_story(genre_or_topic):
    prompt = f"Write a {genre_or_topic} story:"
    response = gpt.Completion.create(
        model=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        n = 1,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# Example usage
specified_genre_or_topic = input("Specify a genre or topic: ")

story = generate_story(specified_genre_or_topic)
print(story)

