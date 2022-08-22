import os 
import sys
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient, __version__

def get_model_names(blob_conn, blob_contain):

    blob_conn_str = blob_conn

    container_name = blob_contain

    storage_client = BlobServiceClient.from_connection_string(blob_conn_str)

    container_client = storage_client.get_container_client(container_name)

    # blob_list = container_client.list_blobs()
    # for blob in blob_list:
    #     print("\t" + blob.name)

    blob_walk = container_client.walk_blobs(delimiter='/')
    model_list = []
    for item in blob_walk:
        model_name = item.name
        model_name = model_name.rstrip('/')
        model_list.append(model_name)

    print(f"Model List:  {model_list}")
        
    return model_list

# if __name__ == "__main__":
#     try:
#         blob_conn_str = os.environ["STORE_CONN_STR"]
#     except ValueError as error:
#         print(error)
#         sys.exit(1)