import os
import time
import pickle
from azure.iot.device import IoTHubModuleClient

class TwinUpdater():

    def __init__(self) -> None:

        try:
            self.client = IoTHubModuleClient.create_from_edge_environment()
            self.client.connect()
            print("Client connected")
            twin_read = self.client.get_twin()
            print(twin_read)
            self.twin_to_config(twin_read)

        except Exception as e:
            print("Unexpected error %s " % e)
            raise

    def twin_to_config(self, twin_raw):
        twin_dict = self.twin_parse(twin_raw)
        config_write = open("/config/variables.pkl", "wb")
        pickle.dump(twin_dict, config_write)
        config_write.close()
        print(f"Config file written")
        self.client.shutdown()

    def twin_parse(self, twin_data):
        twin_variables = {
            "CAMERA_TYPE": twin_data["desired"]["CAMERA_TYPE"],
            "CAMERA_TRIGGER": twin_data["desired"]["CAMERA_TRIGGER"],
            "CAMERA_ID": twin_data["desired"]["CAMERA_ID"],
            "CAMERA_URI": twin_data["desired"]["CAMERA_URI"],
            "CAMERA_LOCATION": twin_data["desired"]["CAMERA_LOCATION"],
            "CAMERA_POSITION": twin_data["desired"]["CAMERA_POSITION"],
            "CAMERA_FPS": twin_data["desired"]["CAMERA_FPS"],
            "INFERENCE_FPS": twin_data["desired"]["INFERENCE_FPS"],
            "MODEL_TYPE": twin_data["desired"]["MODEL_TYPE"],
            "MODEL_NAME": twin_data["desired"]["MODEL_NAME"],
            "MODEL_VERSION": twin_data["desired"]["MODEL_VERSION"],
            "MODEL_ACV_OCR_URI": twin_data["desired"]["MODEL_ACV_OCR_URI"],
            "MODEL_ACV_OCR_SECONDARY": twin_data["desired"]["MODEL_ACV_OCR_SECONDARY"],
            "TARGET_DIM": twin_data["desired"]["TARGET_DIM"] ,
            "PROB_THRES": twin_data["desired"]["PROB_THRES"],
            "IOU_THRES": twin_data["desired"]["IOU_THRES"],
            "RETRAIN_INTERVAL": twin_data["desired"]["RETRAIN_INTERVAL"],
            "STORE_RAW_FRAMES": twin_data["desired"]["STORE_RAW_FRAMES"],
            "STORE_ALL_INFERENCES": twin_data["desired"]["STORE_ALL_INFERENCES"] ,   
        }
        
        return twin_variables
