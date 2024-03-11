from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from crate import client
class ModelConf():
    def __init__(self,device="cpu") -> None:
        self.device=device
    
    def getModel(self):
        return AutoModel.from_pretrained("openai/clip-vit-base-patch16").to(self.device)

    def getProcessor(self):
        return AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    def getTokenizer(self):
        return AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
    
    def getModelConfImg(self):
        model = self.getModel()
        processor = self.getProcessor()
        return model,processor
    
    def getModelConfText(self):
        model = self.getModel()
        tokenizer = self.getTokenizer()
        return model,tokenizer

class CrateConf():
    def __init__(self,host="localhost:4200",username="crate"):
        self.host=host
        self.username=username
    
    def getCursor(self):
        connection = client.connect(self.host, username=self.username)
        cursor = connection.cursor()
        return cursor