from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from crate import client
class ModelConf:
    def __init__(self,device="cpu") -> None:
        self.device=device
    
    def get_model(self):
        return AutoModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)

    def get_processor(self):
        return AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    
    def get_model_conf_img(self):
        model = self.get_model()
        processor = self.get_processor()
        return model,processor
    
    def get_model_conf_text(self):
        model = self.get_model()
        tokenizer = self.get_tokenizer()
        return model,tokenizer

class CrateConf:
    def __init__(self,host="localhost:4200",username="crate"):
        self.host=host
        self.username=username
    
    def get_cursor(self):
        connection = client.connect(self.host, username=self.username)
        cursor = connection.cursor()
        return cursor
