import json
from types import SimpleNamespace
import yaml


class Settings:
    def __init__(self) -> None: 
        self.config: SimpleNamespace = SimpleNamespace()

    def load_config(self, path: str = "./configs/default.yaml") -> None:
        full_path = path
        
        with open(full_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}
            
        self.config = json.loads(
            json.dumps(raw_dict), 
            object_hook=lambda d: SimpleNamespace(**d)
        )
 
settings = Settings()
