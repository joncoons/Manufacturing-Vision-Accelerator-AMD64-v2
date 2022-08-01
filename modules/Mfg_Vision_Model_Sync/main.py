# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for
# full license information.

import time
import os
import sys
import asyncio
from six.moves import input
import threading
from datetime import datetime, timezone
import pathlib 

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, BlobProperties, BlobType, __version__

def main():
    while True:

        blob_service_client = BlobServiceClient.from_connection_string(blob_connstr, api_version='2019-07-07')
        try:
            container_client = blob_service_client.create_container(blob_container)
        except Exception as e:
            pass

        container_client = blob_service_client.get_container_client(blob_container)

        for blob in container_client.list_blobs():
            blob_lastmod = blob.last_modified
            local_path = os.path.join(local_base_dir, blob['name'])
            dir_name = os.path.dirname(local_path)
            dir_path = os.path.expanduser(dir_name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_path)

            if not os.path.isfile(local_path):
                data = container_client.download_blob(blob['name']).readall()
                with open(local_path, 'wb') as file:
                    file.write(data)
            else: 
                local_lastmod = datetime.fromtimestamp(os.stat(local_path).st_mtime, tz=timezone.utc)
                if blob_lastmod > local_lastmod:
                    data = container_client.download_blob(blob['name']).readall()
                    with open(local_path, 'wb') as file:
                        file.write(data)
                    
            # else:
            #     for filepath in pathlib.Path(local_base_dir).rglob('*'):
            #         file_path_list = filepath.parts
            #         if filepath.is_file():
            #             print(filepath.parts)
            #             # print(os.path.basename(local_base_dir))
            #             file_lastmod = datetime.fromtimestamp(os.stat(filepath).st_mtime, tz=timezone.utc)
            #         #     if file_lastmod > blob_lastmod:
                    #         container_client.upload_blob(filepath.name, filepath.open('rb'))
                    #         print(f"Uploaded {filepath.name}")
                    #         blob_lastmod = file_lastmod
                    #         break

                

            

        time.sleep(60)




if __name__ == "__main__":
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(main())
    # loop.close()
    local_base_dir = os.environ['LOCAL_FILE_PATH']
    blob_connstr = os.environ['STORE_CONN_STR']
    blob_container = os.environ['BLOB_AZURE_MODEL_CONTAINER']
    main()

    # If using Python 3.7 or above, you can use following code instead:
    # asyncio.run(main())
