import datetime
import chromadb
import traceback

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

from chromadb.utils import embedding_functions
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
import time
import datetime

def generate_hw01():
    # embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_base = gpt_emb_config['api_base'],
                    api_key = gpt_emb_config['api_key'],
                    api_type = gpt_emb_config['openai_type'],
                    api_version = gpt_emb_config['api_version'],
                    model_name = gpt_emb_config['deployment_name'],
                )


    chroma_client = chromadb.PersistentClient(dbpath) # store in local machine

    # deltet old one first on colab
    try:
      collection = chroma_client.get_collection(name="TRAVEL")
      if (collection.count != 0):
        chroma_client.delete_collection(name="TRAVEL")
    except:
      print("No old collection to be deleted")

    # create chroma
    #collection = chroma_client.get_or_create_collection(
    collection = chroma_client.get_or_create_collection(
        name = "TRAVEL",
        metadata = {
            #使用哪一種評估方式來了解後續詢問的問題，是否接近知識點，我們這邊選擇 Cosine Distance 來作為計算
            "hnsw:space": "cosine"
        },
        embedding_function = openai_ef
    )

    # [Github] Upload chroma.sqlite3, creating from colab, to github
    # [Github] Use the existing collection from chroma.sqlite3  for auto test on github
    if (collection.count != 0):
      return collection

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

    #print(csv_data_name)


    # store csv data into collection from chroma
    size = len(csv_data_name)
    print(size)
    # for deubg
    #for i in range(1): # start from 0
    #    print(csv_data_name[i].metadata["source"])
    #    print(csv_data_name[i])
    #    print(csv_data_createdate[i].metadata["source"])
    #    print(csv_data_createdate[i])
        #print("\n")

    # convert data to list
    documents_metadatas = list()
    documents_hostwords = list()
    for i in range(size): # start from 0
        documents_hostwords.append(
            csv_data_hostwords[i].metadata["source"]
        )

        create_date = csv_data_createdate[i].metadata["source"]
        converted_seec = time.mktime(datetime.datetime.strptime(create_date, "%Y-%m-%d").timetuple())
        if i == 0:
          print("create_date =", create_date, ",is converted to sec = ", converted_seec)

        metadata_string = {
            "file_name" : csv_file_path,
            "name" : csv_data_name[i].metadata["source"],
            "type" : csv_data_type[i].metadata["source"],
            "address" : csv_data_add[i].metadata["source"],
            "tel" : csv_data_tel[i].metadata["source"],
            "city" : csv_data_city[i].metadata["source"],
            "town" : csv_data_town[i].metadata["source"],
            "date" : converted_seec,

        }
        documents_metadatas.append(metadata_string)

    # add to chroma collection
    collection.add(
        ids=[f"{i}" for i in range(size)],
        documents = documents_hostwords,
        metadatas = documents_metadatas)

    return collection
    
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
