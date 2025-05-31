from typing import Optional


class ProviderConfig:
    def __init__(
            self,
            name: str,
            description: Optional[str] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.api_key = api_key
        self.base_url = base_url 