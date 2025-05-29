import os
import logging
from dotenv import load_dotenv

from openai import OpenAI
from typing import Optional

load_dotenv()

# temp logger for now, TODO: to centralize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProviderConfig:
    def __init__(
            self, description: str,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
    ):
        self.description = description
        self.api_key = api_key
        self.base_url = base_url


class LLMClient:
    def __init__(
            self,
            config: ProviderConfig,       
    ):
        self.config = config
        self.client: Optional[OpenAI] = None
        self.init_client()

    def init_client(self):
        try:
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")

    def get_model_list(self):
        try:
            model_list = self.client.models.list()
            models = list(model_list.data)
            for id, model in enumerate(models):
                print(f"{id} - {model.id}")
            return models
        except Exception as e:
            logger.error(f"Error getting model list: {e}")
            return []
        
    def chat_completions(
            self,
            chosen_model,
            input_list,
            stream: bool = True
    ):
        # TODO: monitor if other providers add OpenAI Responses API to their OpenAI-compatibility
        # TODO: add other chat parameters
        try: 
            response = self.client.chat.completions.create(
                model=chosen_model,
                messages=input_list,
                stream=stream,
            )
            return response
        except Exception as e:
            logger.error(f"Error in chat completions: {e}")
            return None
        

if __name__ == "__main__":
    config = ProviderConfig(
        description="Test",
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )

    llm_client = LLMClient(config)
    # present the user with a list of models, to choose for the chat step
    model_list = llm_client.get_model_list()
    
    if not model_list:
        print("No models available or failed to get model list.")
        exit(1)
    
    # get the user's choice, via the index
    is_completed = False
    while not is_completed:
        chosen_model_index = input("Enter the index/number of the model you want to use: ")
        if chosen_model_index.isdigit() and 0 <= int(chosen_model_index) < len(model_list):
            chosen_model = model_list[int(chosen_model_index)].id  # Get the model ID

            user_prompt = input("Enter your prompt: ")
            messages = [{"role": "user", "content": user_prompt}]
            response = llm_client.chat_completions(chosen_model, messages)
            
            if response:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end="", flush=True)
            else:
                print("Failed to get response from the model.")

            is_completed = True
        else:
            print("Invalid input. Please enter a valid index.")
    