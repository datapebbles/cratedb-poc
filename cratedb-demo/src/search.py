#!/usr/bin/env python3
from conf import CrateConf,ModelConf
from PIL import Image

def search_str(prmt):
    #print(prmt)
    model , tokenizer = ModelConf().getModelConfText()
    crateCursor = CrateConf().getCursor()
    text=model.get_text_features(**tokenizer([prmt], return_tensors="pt", truncation=True))
    embedding = text.tolist()[0]
    query = f"SELECT filename FROM retail_data WHERE knn_match(embeddings, {embedding}, 2) ORDER BY _score DESC limit 1"
    crateCursor.execute(query)
    result = crateCursor.fetchall()
    print(result[0])

if __name__ == "__main__":
    input = input("What are you looking for? \n")
    if len(input) > 0:
        search_str(input)
    else:
        print('Acceptable query format: String')