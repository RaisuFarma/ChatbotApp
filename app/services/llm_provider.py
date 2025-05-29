import os
import logging
import asyncio
from dotenv import load_dotenv

from openai import AsyncOpenAI
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
        self.client: Optional[AsyncOpenAI] = None
        self.model_list = []
        self.init_client()

    def init_client(self):
        try:
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            self.client = None

    async def get_model_list(self):
        if not self.client:
            logger.error("Client not initialized")
            return []
        
        try:
            model_list = await self.client.models.list()
            models = list(model_list.data)
            for id, model in enumerate(models):
                print(f"{id} - {model.id}")
            self.model_list = models
            return models
        except Exception as e:
            logger.error(f"Error getting model list: {e}")
            return []
        
    async def chat_completions(
            self,
            model_name: str,
            input_list,
            stream: bool = True
    ):
        if not self.client:
            logger.error("Client not initialized")
            return
        
        try: 
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=input_list,
                stream=stream,
            )

            if stream:
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                yield response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in chat completions: {e}")
        

async def test():
    config = ProviderConfig(
        description="Test",
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )

    llm_client = LLMClient(config)

    model_list = await llm_client.get_model_list()
    
    if not model_list:
        print("No models available or failed to get model list.")
        exit(1)

    chosen_model = None
    while chosen_model is None:
        chosen_model_index = input("Enter the index/number of the model you want to use: ")
        if chosen_model_index.isdigit() and 0 <= int(chosen_model_index) < len(model_list):
            chosen_model = model_list[int(chosen_model_index)].id
        else:
            print("Invalid input. Please enter a valid index.")

    user_prompt = input("Enter your prompt: ")
    
    messages = [{"role": "user", "content": user_prompt}]
    
    print("Non-streaming response:")
    async for chunk in llm_client.chat_completions(chosen_model, messages, False):
        print(chunk)

    print("\nStreaming response:")
    async for chunk in llm_client.chat_completions(chosen_model, messages, True):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(test())
