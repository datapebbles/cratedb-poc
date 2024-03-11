# Image search using CrateDB's Vector matching

This POC is aimed at exploring the vector search capabitilies of CrateDB.
For this use case we will be using a Multi modal algorithm from HuggingFace (https://huggingface.co/learn/cookbook/en/faiss_with_hf_datasets_and_clip) to generate embeddings, store and query in cratedb as a vector.

### The repo contains below three files.
>
#### conf.py 
> - This python script contains the definitions for of configurations of the Multi modal ML algorithm being used and CrateDB.
>
#### train.py
> - This python script takes two inputs 
>      1. The folder which contains the images dataset whose vectors are to be generated
>      2. The folder where the images should be moved after processing, to avoid reprocessing in case of failures.
>       
> - The script processes the files in source folder, generates embeddings and stores them in a table retail_data as a vector.
>
#### search.py
> - This python script takes one input, which is the search term and returns the closest matching image from the table retail_data.
> - Internally this table generates embedding from the text image and queries the database using KNN matching function of cratedb.

<br>

# Note:
- For Cratedb I have a single node setup locally on docker and port-forward to 4200
>       docker run --publish 4200:4200 --publish 5432:5432 --env CRATE_HEAP_SIZE=1g crate -Cdiscovery.type=single-node
<br>

- For python, I am using a conda environment with python 3.9 and have listed the requirements in **requirements.txt**
- In train.py while storing the vector, we are also storing the file name along with the path. However the path is no longer valid as we are moving the file to a different folder.
- Below is the DDL for creating the table in cratedb.
> -   CREATE TABLE retail_data (
  filename STRING,  
  embeddings FLOAT_VECTOR(512)
);