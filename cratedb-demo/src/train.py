#!/usr/bin/env python3
from conf import ModelConf,CrateConf
import argparse
import sys
import os
from PIL import Image

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_args(args):
    parser = argparse.ArgumentParser(description="Generate embeddings for images")
    parser.add_argument(
        '--folder-path',
        help="Folder with image dataset which should be embedded",
        required=True
    )
    parser.add_argument(
        '--processed-path',
        help="Folder where image should be moved to after processing",
        required=True
    )
    return parser.parse_args(args)

'''def getModelConf():
    modelConf = ModelConf()
    model = modelConf.getModel()
    processor = modelConf.getProcessor()
    return model,processor'''


def main(args):
    parsed_args = parse_args(args)
    kwargs = vars(parsed_args)
    print(kwargs)
    files_loc = kwargs['folder_path']
    processed_files = kwargs['processed_path']
    create_folder(processed_files)
    if os.path.isdir(files_loc):
        files = os.listdir(files_loc)
        model , processor = ModelConf.getModelConfImg()
        crateCursor = CrateConf().getCursor()
        results = []
        counter = 0
        file_counter = 1
        for file in files:
            file_name_path = files_loc+'/'+file
            print(f"processing file no :{file_counter} , name:{file}")
            image = Image.open(file_name_path)
            embedding = model.get_image_features(**processor(image, return_tensors="pt"))
            results.append((file_name_path, embedding.tolist()[0]))
            image.close()
            counter = counter + 1
            file_counter += 1
            os.rename(file_name_path, f"{processed_files}/{file}")
            if counter == 10:
                print(results)
                crateCursor.executemany("insert into retail_data (filename,embeddings) values (?,?)",results)
                print("inserted batch of 10 file embeddings")
                del(results)
                results = []
                counter = 0
        if len(results) > 0:
            crateCursor.executemany("insert into retail_data (filename,embeddings) values (?,?)",results) 
        crateCursor.close()
    else:
        raise Exception(f"{files_loc} Not a directory")


if __name__ == "__main__":
    main(sys.argv[1:])