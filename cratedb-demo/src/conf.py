from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from crate import client
class ModelConf:
    def __init__(self,device="cpu") -> None:
        self.device=device
        self.model = AutoModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)
        self.processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")

class CrateConf:
    def __init__(self,host="localhost:4200",username="crate") -> None:
        self.host=host
        self.username=username
    
    def get_cursor(self):
        connection = client.connect(self.host, username=self.username)
        cursor = connection.cursor()
        return cursor
