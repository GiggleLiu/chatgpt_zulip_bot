# chatgpt.py
import openai
from configparser import ConfigParser
import tiktoken

config = ConfigParser()
config.read("config.ini")
OPENAI_API_KEY = config["settings"]["OPENAI_API_KEY"]
OPENAI_API_VERSION = config["settings"]["API_VERSION"]
openai.api_key = OPENAI_API_KEY

user_conversations = {}  # Maintain a dictionary to store conversation history per user
MAX_CONTENT_LENGTH = 4097 - 300


def trim_conversation_history(history, max_tokens):
    # Determine the appropriate encoding based on the API version
    if "gpt-3.5-turbo" in OPENAI_API_VERSION:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    elif "gpt-4" in OPENAI_API_VERSION:
        encoding = tiktoken.encoding_for_model("gpt-4")
    elif 'text-embedding-ada' in OPENAI_API_VERSION:
        encoding = tiktoken.encoding_for_model('text-embedding-ada-002')
    elif "text-davinci-002" in OPENAI_API_KEY:
        encoding = tiktoken.encoding_for_model("text-davinci-002")
    elif 'text-davinci-003' in OPENAI_API_KEY:
        encoding = tiktoken.encoding_for_model("text-davinci-003")
    else:
        return "OpenAI API Version Wrong!"

    tokens = 0
    trimmed_history = []
    for message in reversed(history):
        # Get the number of tokens for the current message
        message_tokens = len(encoding.encode(message))
        if tokens + message_tokens <= max_tokens:
            trimmed_history.insert(0, message)
            tokens += message_tokens
        else:
            break

    return trimmed_history



def get_chatgpt_response(user_id, prompt):
    global user_conversations

    if user_id not in user_conversations:
        user_conversations[
            user_id
        ] = []  # Create a new conversation history for a new user

    # Check if user input is "停止会话" or "end the conversation"
    if prompt == "停止会话" or prompt.lower() == "end the conversation" or prompt.lower() == '/end':
        user_conversations[user_id] = []  # Clear the conversation history for the user
        return "The conversation has been ended and the context has been cleared."

    conversation_history = user_conversations[user_id]
    conversation_history.append(
        f"User: {prompt}"
    )  # Add user input to conversation history

    while True:
        messages = [
            {
                "role": "system",
                "content": "You are an AI language model trained to assist with a variety of tasks.",
            }
        ]  # System message for context

        for message in conversation_history:
            role, content = message.split(": ", 1)
            messages.append({"role": role.lower(), "content": content})

        try:
            response = openai.ChatCompletion.create(
                model=OPENAI_API_VERSION,
                messages=messages,
                max_tokens=1200,
                temperature=0.5,
            )

            if response.choices:
                role = response["choices"][0]["message"]["role"]
                reply = (
                    response["choices"][0]["message"]["content"].strip().replace("", "")
                )
                conversation_history.append(
                    f"{role}: {reply}"
                )  # Add AI response to conversation history
                return (
                    reply
                    + "\n------\nPrompt tokens used: "
                    + str(response.usage.prompt_tokens)
                    + "\nAnswer tokens used: "
                    + str(response.usage.completion_tokens)
                    + "\nTotal tokens used: "
                    + str(response.usage.total_tokens)
                )
            else:
                return "Sorry, I couldn't generate a response."

        except openai.error.RateLimitError:
            print("ERROR: OpenAI API rate limit exceeded. Please retry.")
            return "ERROR: OpenAI API rate limit exceeded. Please retry."

        except openai.error.OpenAIError as e:
            if "Please reduce the length" in str(e):
                conversation_history = trim_conversation_history(
                    conversation_history, MAX_CONTENT_LENGTH
                )
            else:
                print(f"Error: {e}")
                return "Sorry, there was an error generating a response."

        except Exception as e:
            print(f"Error: {e}")
            return "Sorry, there was an error generating a response."
