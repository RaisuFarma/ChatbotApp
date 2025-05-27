from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")


client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)


def list_models(client):
    return client.models.list()


def chat_completions(client, model: str, messages:list):
    return client.chat.completions.create(
        model=model,
        messages=messages,
    )


# Test
if __name__ == "__main__":
    model_list = list_models(client)
    for id, model in enumerate(model_list):
        print(f'{id} - {model.id}')

    response = chat_completions(client, "Doctor-Shotgun-L3.3-70B-Magnum-v4-SE", [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Are tarrifs good for the economy?"},
    ])

    print(response)

