from markupsafe import escape
from azure.cosmos import CosmosClient, PartitionKey, exceptions
# import azure.cosmos.cosmos_client as cosmos_client
# import azure.cosmos.exceptions as exceptions
# from azure.cosmos.partition_key import PartitionKey

# p_key = PartitionKey(path='/deviceId', kind='Hash')

def cosmos_connect(db_name, uri, key, collection):
    client = CosmosClient(uri, credential=key, consistency_level='Session')

    # db = client.create_database_if_not_exists(id=db_name)
    try:
        db = client.create_database(db_name)
    except exceptions.CosmosResourceExistsError:
        db = client.get_database_client(db_name)
    try:
        container = db.create_container(id=collection, partition_key=PartitionKey(path="/deviceId"))
    except exceptions.CosmosResourceExistsError:
        container = db.get_container_client(collection)
    return container

def cosmos_delete_db(db_name, uri, key):
    client = CosmosClient(uri, credential=key)
    try:
        client.delete_database(db_name)
        print('Database with id \'{0}\' was deleted'.format(db_name))
    except exceptions.CosmosResourceNotFoundError:
        print('A database with id \'{0}\' does not exist'.format(db_name))

def cosmos_delete_items(cdb_col_name):
    doc_list = list(cdb_col_name.read_all_items())
    for doc in doc_list:
        cdb_col_name.delete_item(item=doc, partition_key=doc)
     
def cosmos_create_items(db_name,uri,key,collection,item):
    container = cosmos_connect(db_name,uri,key,collection)
    doc_list = list(container.read_all_items())
    try:
        response = container.create_item(body=item)
        update_type = "created"
    except:
        response = container.upsert_item(body=item)
        update_type = "upserted"
    return update_type

def cosmos_query_dm(db_name, uri, key, collection):
    container = cosmos_connect(db_name,uri,key,collection)
    d_list = list(container.query_items(
        query=f"SELECT DISTINCT r.deviceId FROM {collection} as r ORDER BY r.deviceId ASC",
        enable_cross_partition_query=True
    ))
    m_list = list(container.query_items(
        query=f"SELECT DISTINCT r.moduleId FROM {collection} as r ORDER BY r.moduleId ASC",
        enable_cross_partition_query=True
    ))
    # device_count = len(d_list)
    # module_count = len(m_list)
    dm_list = list(container.query_items(
        query=f"SELECT r.deviceId, r.moduleId FROM {collection} as r ORDER BY r.moduleId ASC",
        enable_cross_partition_query=True
    ))
    return (d_list,m_list,dm_list)

def cosmos_query_mt(db_name, uri, key, collection, device, module):
    container = cosmos_connect(db_name,uri,key,collection)
    item_id = f"{device}-{module}"
    m_query = list(container.query_items(
        query=f"SELECT * FROM {collection} as r WHERE r.id=@id",
        parameters=[
            { "name":"@id", "value": item_id }
        ],
        enable_cross_partition_query=True
    ))
    m_count = len(m_query)
    print(f"Query Count: {m_count}")
    # print(m_query)
    return m_query