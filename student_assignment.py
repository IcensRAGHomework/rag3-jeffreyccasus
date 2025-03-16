import datetime
import chromadb
import traceback

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document

def generate_hw01():
    # load csv
    csv_file_path = "COA_OpenData.csv"
    csv_data_name = CSVLoader(file_path = csv_file_path, source_column="Name").load()
    csv_data_type = CSVLoader(file_path = csv_file_path, source_column="Type").load()
    csv_data_add = CSVLoader(file_path = csv_file_path, source_column="Address").load()
    csv_data_tel = CSVLoader(file_path = csv_file_path, source_column="Tel").load()
    csv_data_city = CSVLoader(file_path = csv_file_path, source_column="City").load()
    csv_data_town = CSVLoader(file_path = csv_file_path, source_column="Town").load()
    csv_data_createdate = CSVLoader(file_path = csv_file_path, source_column="CreateDate").load()
    csv_data_hostwords = CSVLoader(file_path = csv_file_path, source_column="HostWords").load()

    #print_csv_detail(csv_data)


    # create chroma
    chroma_client = chromadb.PersistentClient(dbpath)
    #collection = chroma_client.get_or_create_collection(
    collection = chroma_client.get_or_create_collection(
        name = "TRAVEL",
        metadata = {
            "hnsw:space": "cosine"
        }
    )

    if collection.count() !=0:
        return collection

    # store csv data into collection from chroma
    size = len(csv_data_name)
    print(size)
    #for i in range(1): # start from 0
        #print(csv_data_hostwords[i].metadata["source"])
        #print(csv_data_name)
        #print("\n")

    documents_metadatas = list()
    documents_hostwords = list()
    for i in range(size): # start from 0
        documents_hostwords.append(
            csv_data_hostwords[i].metadata["source"]
        )
        metadata_string = {
            "file_name" : csv_file_path,
            "name" : csv_data_name[i].metadata["source"],
            "type" : csv_data_type[i].metadata["source"],
            "address" : csv_data_add[i].metadata["source"],
            "tel" : csv_data_tel[i].metadata["source"],
            "city" : csv_data_city[i].metadata["source"],
            "town" : csv_data_town[i].metadata["source"],
            "date" : csv_data_createdate[i].metadata["source"],

        }
        documents_metadatas.append(metadata_string)


    collection.add(
        ids=[f"{i}" for i in range(size)],
        documents = documents_hostwords,
        metadatas = documents_metadatas)

    return
    
def generate_hw02(question, city, store_type, start_date, end_date):
    pass
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection
