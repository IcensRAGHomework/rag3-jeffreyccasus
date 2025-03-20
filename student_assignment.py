import datetime
import chromadb
import traceback

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

import chromadb.utils.embedding_functions as embedding_functions

import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
import time
import datetime

def get_embedding_function():
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_base = gpt_emb_config['api_base'],
                    api_key = gpt_emb_config['api_key'],
                    api_type = gpt_emb_config['openai_type'],
                    api_version = gpt_emb_config['api_version'],
                    model_name = gpt_emb_config['deployment_name'],
                )
    
    #openai_ef = embedding_functions.DefaultEmbeddingFunction()  
    return openai_ef
local_collection_name = "TRAVEL"

def is_collection_existing(colletion_name):
    collection_existing = False

    persistent_client = chromadb.PersistentClient(dbpath) # store in local machine

    # 取得所有 collection 名稱
    existing_collections_name = persistent_client.list_collections()

    # 檢查是否存在特定 collection
    
    if colletion_name in existing_collections_name:
        print(f"Collection '{colletion_name}' 已存在")
        collection_existing = True
    else:
        print(f"Collection '{colletion_name}' 不存在")
        collection_existing = False

    return collection_existing

def delete_old_collection():
    if is_collection_existing(local_collection_name):
      persistent_client = chromadb.PersistentClient(dbpath)
      persistent_client.delete_collection(local_collection_name)
      print(f"刪除 collection '{local_collection_name}' !! ")
    return

def generate_hw01():
    # embedding function
    openai_ef = get_embedding_function()

    persistent_client = chromadb.PersistentClient(dbpath) # store in local machine

    # 取得所有 collection 名稱
    existing_collections_name = persistent_client.list_collections()

    # 檢查是否存在特定 collection
    collection_existing = False
    if is_collection_existing(local_collection_name):
      collection_existing = True

    # create chroma
    collection = persistent_client.get_or_create_collection(
        name = local_collection_name,
        metadata = {
            #使用哪一種評估方式來了解後續詢問的問題，是否接近知識點，我們這邊選擇 Cosine Distance 來作為計算
            "hnsw:space": "cosine"
        },
        embedding_function = openai_ef
    )

    if (collection_existing):
        return collection


    # load csv
    csv_file_path = "COA_OpenData.csv"
    df = pd.read_csv(csv_file_path)

    csv_data_name = df["Name"].astype(str).tolist()
    csv_data_type = df["Type"].astype(str).tolist()
    csv_data_add = df["Address"].astype(str).tolist()
    csv_data_tel = df["Tel"].astype(str).tolist()
    csv_data_city = df["City"].astype(str).tolist()
    csv_data_town = df["Town"].astype(str).tolist()
    csv_data_createdate = df["CreateDate"].astype(str).tolist()
    csv_data_hostwords = df["HostWords"].astype(str).tolist()
    #print(csv_data_name)


    # store csv data into collection from chroma
    size = len(csv_data_name)
    print(f"csv data size =  '{size}'")
    # for deubg
    #for i in range(1): # start from 0
    #    print(csv_data_name[i].metadata["source"])
    #    print(csv_data_name[i])
    #    print(csv_data_createdate[i].metadata["source"])
    #    print(csv_data_createdate[i])
        #print("\n")

    # convert data to list
    documents_metadatas = list()
    for i in range(size): # start from 0
        create_date = csv_data_createdate[i]
        converted_seec = time.mktime(datetime.datetime.strptime(create_date, "%Y-%m-%d").timetuple())
        if i == 0:
          print("create_date =", create_date, ",is converted to sec = ", converted_seec)

        metadata_string = {
            "file_name" : csv_file_path,
            "name" : csv_data_name[i],
            "type" : csv_data_type[i],
            "address" : csv_data_add[i],
            "tel" : csv_data_tel[i],
            "city" : csv_data_city[i],
            "town" : csv_data_town[i],
            "date" : converted_seec,

        }
        documents_metadatas.append(metadata_string)

    # add to chroma collection and save data to chroma.sqlite3!
    collection.add(
        ids = [f"{i}" for i in range(size)],
        documents = csv_data_hostwords,
        metadatas = documents_metadatas)

    return collection

    
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import datetime

def generate_hw02(question, city, store_type, start_date, end_date):
    max_result_number = 10
    similarity_threshold = 0.80

    start_date_sec = start_date.timestamp()
    end_date_sec = end_date.timestamp()
    #print("start_date_sec = ", start_date_sec)
    #print("end_date_sec = ", end_date_sec)

    collection = generate_hw01()

    # Chroma metadata filter, https://docs.trychroma.com/docs/querying-collections/metadata-filtering
    # filter collection by requirement first
    '''
    filter_dict  = {
        "$and": [
            {
                "city" : {"$in": city}
            },
            {
                "type": {"$in": store_type}
            },
            {
                "date": {"$gte": start_date_sec} # $gte - greater than or equal to (int, float)
            },
            {
                "date": {"$lte": end_date_sec} # $lte - less than or equal to (int, float)
            }
        ]
    }
    '''

    filter_condition = list()
    temp = ""

    if (len(city) != 0):
        temp = {"city" : {"$in": city}}
        filter_condition.append(temp)
        
    if (len(store_type) != 0):
        temp = {"type": {"$in": store_type}}
        filter_condition.append(temp)
    
    temp =  {
              "date": {"$gte": start_date_sec} # $gte - greater than or equal to (int, float)
            }
    filter_condition.append(temp)

    temp = {
              "date": {"$lte": end_date_sec} # $lte - less than or equal to (int, float)
           }
    filter_condition.append(temp)
    
    filter_dict  = {"$and": filter_condition}
    print(f"filter_dict = '{filter_dict}' \n ")

 
    # https://docs.trychroma.com/docs/querying-collections/query-and-get
    results = collection.query(
        query_texts = question, # Chroma will embed this for you
        include = ["documents", "metadatas", "distances"],
        n_results = max_result_number,
        where = filter_dict,
    )
    print(f"query result = '{results}' \n ")

    # 提取結果 (包含 id, metadata, score 等資訊), 將結果按照 distance 排序
    sorted_results = sorted(
        zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]),
        key=lambda x: x[3],  # 依照 distance 排序 (越小越相似)
        reverse=False  # 遞增排序，確保最相似的在前
    )

    # 過濾 score (確保相似度高於 0.8，通常距離越小代表相似度越高)
    filtered_results = [(id, doc, meta, score) for id, doc, meta, score in sorted_results if score <= (1 - similarity_threshold)]

    # 輸出結果
    for id, doc, meta, score in filtered_results:
        print(f"ID: {id}, Score: {1 - score:.4f}\nDocument: {doc}\nMetadata: {meta}\n")

    # 將過濾 score 後的結果, 提取 metadata 中 店家名稱 為獨立 list
    meta_list = [meta.get("name") for _, _, meta, _ in filtered_results]

    # 輸出結果
    for id, doc, meta, score in filtered_results:
        print(f"ID: {id}, Score: {1 - score:.4f}\nDocument: {doc}\nMetadata: {meta}\n")

    return meta_list
    
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
