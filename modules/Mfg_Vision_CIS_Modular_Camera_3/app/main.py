import os
import pickle
from typing import Any, Union
from threading import Thread
from time import sleep
from twin_call import TwinUpdater
from azure.iot.device import IoTHubModuleClient, Message

class HubConnector():

    def __init__(self):
        self.client = IoTHubModuleClient.create_from_edge_environment()
        # self.client = IoTHubModuleClient.create_from_connection_string(os.environ["MODULE_CONN_STR"])
        self.client.connect()

    def send_to_output(self, message: Union[Message, str], outputQueueName: str):
        self.client.send_message_to_output(message, outputQueueName)

def send_to_upload(msg_str: str) -> None:
    message = Message(bytearray(msg_str, 'utf-8'))
    hub_connector.send_to_output(message, "outputImageSend")

def send_to_upstream(msg_str: str) -> None:
    message = Message(bytearray(msg_str, 'utf-8'))
    hub_connector.send_to_output(message, "outputInference")

class CaptureInferenceStore():

    def __init__(self, camGvspAllied, camGvspBasler, camRTSP, camFile, camID, camTrigger, camURI, camLocation, camPosition, camFPS, inferenceFPS, 
        modelAcvOD, modelAcvMultiClass, modelAcvMultiLabel, modelAcvOcr, modelAcvOcrUri, modelAcvOcrSecondary, modelYolov5, modelFasterRCNN, modelRetinanet, modelMaskRCNN, modelClassMultiLabel, modelClassMultiClass, 
        modelName, modelVersion, targetDim, probThres, iouThres, retrainInterval, storeRawFrames, storeAllInferences,):

        modelPath = f'/model_volume/{modelName}/{modelVersion}/model.onnx'
        labelPath = f'/model_volume/{modelName}/labels.json'
        modelFile = f'{modelName}-v.{modelVersion}'
        labelFile = 'labels.json'

        sleep(3)

        if modelAcvOD:
            from inference.ort_acv_predict import initialize_acv
            labelPath = f'/model_volume/{modelName}/labels.txt'
            labelFile = 'labels.txt'
            initialize_acv(modelPath, labelPath)
        elif modelAcvMultiClass:
            from inference.ort_acv_mc_class import initialize_acv_mc_class
            labelPath = f'/model_volume/{modelName}/labels.txt'
            labelFile = 'labels.txt'
            initialize_acv_mc_class(modelPath, labelPath)
        elif modelAcvMultiLabel:
            from inference.ort_acv_ml_class import initialize_acv_ml_class
            labelPath = f'/model_volume/{modelName}/labels.txt'
            labelFile = 'labels.txt'
            initialize_acv_ml_class(modelPath, labelPath)
        elif modelYolov5:
            from inference.ort_yolov5 import initialize_yolov5
            initialize_yolov5(modelPath, labelPath, targetDim, probThres, iouThres)
        elif modelFasterRCNN:
            from inference.ort_faster_rcnn import initialize_faster_rcnn
            initialize_faster_rcnn(modelPath, labelPath, targetDim, probThres, iouThres)
        elif modelRetinanet:
            from inference.ort_retinanet import initialize_retinanet
            initialize_retinanet(modelPath, labelPath, targetDim, probThres, iouThres)
        elif modelMaskRCNN:
            from inference.ort_mask_rcnn import initialize_mask_rcnn
            initialize_mask_rcnn(modelPath, labelPath, targetDim, probThres, iouThres)
        elif modelClassMultiLabel:
            from inference.ort_class_multi_label import initialize_class_multi_label
            initialize_class_multi_label(modelPath, labelPath, targetDim, probThres, iouThres)
        elif modelClassMultiClass:
            from inference.ort_class_multi_class import initialize_class_multi_class
            initialize_class_multi_class(modelPath, labelPath, targetDim, probThres, iouThres)
        else:
            print('No model selected')

        sleep(1)

        if camGvspAllied:     
            from capture.allied.camera_gvsp_allied import Allied_GVSP_Camera
            Allied_GVSP_Camera(camID, camTrigger, camURI, camLocation, camPosition, camFPS, inferenceFPS, 
            modelAcvOD, modelAcvMultiClass, modelAcvMultiLabel, modelAcvOcr, modelAcvOcrUri, modelAcvOcrSecondary, 
            modelYolov5, modelFasterRCNN, modelRetinanet, modelMaskRCNN, modelClassMultiLabel, modelClassMultiClass, 
            modelName, modelVersion, targetDim, probThres, iouThres, retrainInterval, storeRawFrames, storeAllInferences, 
            modelFile, labelFile, send_to_upload, send_to_upstream)

        if camGvspBasler:     
            from capture.basler.camera_gvsp_basler import Basler_GVSP_Camera
            Basler_GVSP_Camera(camID, camTrigger, camURI, camLocation, camPosition, camFPS, inferenceFPS, 
            modelAcvOD, modelAcvMultiClass, modelAcvMultiLabel, modelAcvOcr, modelAcvOcrUri, modelAcvOcrSecondary, 
            modelYolov5, modelFasterRCNN, modelRetinanet, modelMaskRCNN, modelClassMultiLabel, modelClassMultiClass, 
            modelName, modelVersion, targetDim, probThres, iouThres, retrainInterval, storeRawFrames, storeAllInferences, 
            modelFile, labelFile, send_to_upload, send_to_upstream)
            
        elif camRTSP:
            from capture.RTSP.camera_rtsp import RTSP_Camera
            RTSP_Camera(camID, camTrigger, camURI, camLocation, camPosition, camFPS, inferenceFPS, 
            modelAcvOD, modelAcvMultiClass, modelAcvMultiLabel, modelAcvOcr, modelAcvOcrUri, modelAcvOcrSecondary, 
            modelYolov5, modelFasterRCNN, modelRetinanet, modelMaskRCNN, modelClassMultiLabel, modelClassMultiClass, 
            modelName, modelVersion, targetDim, probThres, iouThres, retrainInterval, storeRawFrames, storeAllInferences, 
            modelFile, labelFile, send_to_upload, send_to_upstream)

        elif camFile:
            from capture.file.camera_file import Cam_File_Sink
            Cam_File_Sink(camID, camTrigger, camURI, camLocation, camPosition, camFPS, inferenceFPS, 
            modelAcvOD, modelAcvMultiClass, modelAcvMultiLabel, modelAcvOcr, modelAcvOcrUri, modelAcvOcrSecondary, 
            modelYolov5, modelFasterRCNN, modelRetinanet, modelMaskRCNN, modelClassMultiLabel, modelClassMultiClass, 
            modelName, modelVersion, targetDim, probThres, iouThres, retrainInterval, storeRawFrames, storeAllInferences, 
            modelFile, labelFile, send_to_upload, send_to_upstream)

        else:
            print("No camera selected")

hub_connector: HubConnector = None

def __convertStringToBool(env: str) -> bool:
    if env in ['true', 'True', 'TRUE', '1', 'y', 'YES', 'Y', 'Yes']:
        return True
    elif env in ['false', 'False', 'FALSE', '0', 'n', 'NO', 'N', 'No']:
        return False
    else:
        raise ValueError('Could not convert string to bool.')

def run_CIS():

    global hub_connector
    try:
        hub_connector = HubConnector()
    except Exception as err:
        print(f"Unexpected error {err} from IoTHub")

    config_read = open("/config/variables.pkl", "rb")
    variables = pickle.load(config_read)
    print(f"Variables: \n{variables}")

    if variables["CAMERA_FPS"]:
        CAMERA_FPS = float(variables["CAMERA_FPS"])
    else: 
        CAMERA_FPS = float(1)

    if variables["INFERENCE_FPS"]:
        INFERENCE_FPS = float(variables["INFERENCE_FPS"])
    else: 
        INFERENCE_FPS = float(1)

    # for ACV models
    os.environ["IOU_THRES"] = variables["IOU_THRES"]
    os.environ['TARGET_DIM'] = variables["TARGET_DIM"]
    os.environ["PROB_THRES"] = variables["PROB_THRES"]

    camera_type = variables["CAMERA_TYPE"]
    camGvspAllied_value = False
    camGvspBasler_value = False
    camRTSP_value = False
    camFile_value = False
    if camera_type == 'Allied Vision GVSP':
        camGvspAllied_value = True
    elif camera_type == 'Basler GVSP':
        camGvspBasler_value = True
    elif camera_type == 'RTSP Camera':
        camRTSP_value = True
    elif camera_type == 'Read from file':
        camFile_value = True


    model_type = variables["MODEL_TYPE"]
    modelAcvOD_value = False
    modelAcvMultiClass_value = False
    modelAcvMultiLabel_value = False
    modelAcvOcr_value = False
    modelYolov5_value = False
    modelFasterRCNN_value = False
    modelRetinanet_value = False 
    modelClassMultiLabel_value = False
    modelClassMultiClass_value = False
    modelMaskRCNN_value = False
    if model_type == 'Azure Custom Vision - Object Detection':
        modelAcvOD_value = True
    elif model_type == 'Azure Custom Vision - Multi-Class Classification':
        modelAcvMultiClass_value = True
    elif model_type == 'Azure Custom Vision - Multi-Label Classification':
        modelAcvMultiLabel_value = True
    elif model_type == 'Azure Computer Vision - OCR Read':
        modelAcvOcr_value = True
    elif model_type == 'AutoML for Images - YOLOv5':
        modelYolov5_value = True
    elif model_type == 'AutoML for Images - Faster-RCNN':
        modelFasterRCNN_value = True
    elif model_type == 'AutoML for Images - Retinanet':
        modelRetinanet_value = True
    elif model_type == 'AutoML for Images - Multi-Label Classification':
        modelClassMultiLabel_value = True
    elif model_type == 'AutoML for Images - Multi-Class Classification':
        modelClassMultiClass_value = True
    elif model_type == 'AutoML for Images - Mask-RCNN':
        modelMaskRCNN_value = True

    CaptureInferenceStore(
        camGvspAllied = camGvspAllied_value, 
        camGvspBasler = camGvspBasler_value,
        camRTSP = camRTSP_value, 
        camFile = camFile_value, 
        camID = variables["CAMERA_ID"],
        camTrigger = variables["CAMERA_TRIGGER"], 
        camURI = variables["CAMERA_URI"], 
        camLocation = variables["CAMERA_LOCATION"], 
        camPosition = variables["CAMERA_POSITION"], 
        camFPS = CAMERA_FPS, 
        inferenceFPS = INFERENCE_FPS, 
        modelAcvOD = modelAcvOD_value,
        modelAcvMultiClass = modelAcvMultiClass_value,
        modelAcvMultiLabel = modelAcvMultiLabel_value,
        modelAcvOcr = modelAcvOcr_value,
        modelAcvOcrUri = variables["MODEL_ACV_OCR_URI"],
        modelAcvOcrSecondary = variables["MODEL_ACV_OCR_SECONDARY"],
        modelYolov5 = modelYolov5_value,
        modelFasterRCNN = modelFasterRCNN_value,
        modelRetinanet = modelRetinanet_value,
        modelClassMultiLabel = modelClassMultiLabel_value,
        modelClassMultiClass = modelClassMultiClass_value,
        modelMaskRCNN = modelMaskRCNN_value,
        modelName = variables["MODEL_NAME"], 
        modelVersion = variables["MODEL_VERSION"], 
        targetDim = int(variables["TARGET_DIM"]), 
        probThres = float(variables["PROB_THRES"]), 
        iouThres = float(variables["IOU_THRES"]), 
        retrainInterval = int(variables["RETRAIN_INTERVAL"]), 
        storeRawFrames = variables["STORE_RAW_FRAMES"], 
        storeAllInferences = variables["STORE_ALL_INFERENCES"], 
        )
        
def twin_update():
    TwinUpdater()

thread1 = Thread(name='twin_update',target=twin_update)
thread2 = Thread(name='run_CIS', target=run_CIS)

if __name__ == "__main__":
    thread1.start()
    thread1.join()
    # sleep(5)
    thread2.start()

    
