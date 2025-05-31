import os
import logging
import time
import asyncio
from dotenv import load_dotenv

from openai import AsyncOpenAI
from typing import Optional, AsyncGenerator

from llm_config import ProviderConfig

load_dotenv()

# temp logger for now, TODO: to centralize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMClient():
    def __init__(
            self,
            config: ProviderConfig,
    ) -> None:
        self.config = config
        self._model_cache_time = None
        self._model_cache_list = None
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    async def close(self) -> None:
        await self.client.aclose()

    def _is_within_cache_ttl(self) -> bool:
        if self._model_cache_time is None or self._model_cache_time <= 0:
            return False

        return (time.time() - self._model_cache_time) < self._model_cache_ttl
    
    async def list_models(
            self,
            refresh: bool = False
    ) -> list:
        if not self.client:
            logger.error("Client not initialized")
            return []
        
        if not refresh and self._is_within_cache_ttl():
            return self._model_cache_list
        
        try:
            model_list = await self.client.models.list()
            models = list(model_list.data)
            for id, model in enumerate(models):
                print(f"{id} - {model.id}")
            self._model_cache_list = models
            self._model_cache_time = time.time()
            return models
        except Exception as e:
            logger.error(f"Error getting model list: {e}")
            return []
        
    async def chat_completions(
            self,
            model: str,
            messages: list,
            stream: bool = True
    ) -> AsyncGenerator[str, None]:
        if not self.client:
            logger.error("Client not initialized")
            return
        
        try: 
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
            )

            if stream:
                async for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
            else:
                yield response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in chat completions: {e}")
        

async def test():
    config = ProviderConfig(
        name="Test",
        description="Test",
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )

    llm_client = LLMClient(config)

    model_list = await llm_client.list_models()
    
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
