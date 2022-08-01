import os
import json
import sys
import time
import uuid
import requests
from io import BytesIO
from PIL import Image
from time import sleep
from typing import Any, Callable, Optional
import cv2
import numpy as np
from datetime import datetime
from capture.allied.vimba import *
from capture.frame_preprocess import frame_resize
from capture.frame_save import FrameSave
from store.sql_insert import InsertInference

capturing = False

class Allied_GVSP_Camera:
    sql_state = 0

    def __init__(
        self, camID, camTrigger, camURI, camLocation, camPosition, camFPS, inferenceFPS, 
        modelAcvOD, modelAcvMultiClass, modelAcvMultiLabel, modelAcvOcr, modelAcvOcrUri, modelAcvOcrSecondary, 
        modelYolov5, modelFasterRCNN, modelRetinanet, modelMaskRCNN, modelClassMultiLabel, modelClassMultiClass, 
        modelName, modelVersion, targetDim, probThres, iouThres, retrainInterval, storeRawFrames, storeAllInferences, 
        modelFile, labelFile, send_to_upload: Callable[[str], None], send_to_upstream: Callable[[str], None]
        ):

        self.camID = camID
        self.camTrigger = camTrigger
        self.camURI = camURI
        self.camLocation = camLocation
        self.camPosition = camPosition
        self.camFPS = camFPS
        self.inferenceFPS = inferenceFPS
        self.modelAcvOcr = modelAcvOcr
        self.modelAcvOcrUri = modelAcvOcrUri
        self.modelAcvOcrSecondary = modelAcvOcrSecondary
        self.modelAcvOD = modelAcvOD
        self.modelAcvMultiClass = modelAcvMultiClass
        self.modelAcvMultiLabel = modelAcvMultiLabel
        self.modelYolov5 = modelYolov5
        self.modelFasterRCNN = modelFasterRCNN
        self.modelRetinanet = modelRetinanet
        self.modelMaskRCNN = modelMaskRCNN
        self.modelClassMultiLabel = modelClassMultiLabel
        self.modelClassMultiClass = modelClassMultiClass
        self.modelFile = modelFile
        self.labelFile = labelFile
        self.targetDim = targetDim
        self.probThres = probThres
        self.iouThres = iouThres
        self.retrainInterval = retrainInterval
        self.storeRawFrames = storeRawFrames
        self.storeAllInferences = storeAllInferences
        self.model_name = modelName
        self.model_version = modelVersion
        self.send_to_upload = send_to_upload
        self.send_to_upstream = send_to_upstream

        self.frameCount = 0
        self.frameRateCount = 0
        self.reconnectCam = True

        self.cycle_begin = 0
        self.cycle_end = 0
        self.t_full_cycle = 0

        self.streamCap()

    def streamCap(self):

        try:
            global capturing
            capturing = False

            def camera_change_handler(dev: Any, state: Any):
                global capturing
                msg = f"Device: {dev}, State: {state}"
                print(msg)
                if state == 2 or state == 3:
                    print("Exiting camera device...")
                    dev.__exit__(None, None, None)
                    capturing = False

            while True:
                # Wait for the camera to be found
                increment = 0
                while not self.check_camera_exists(self.camID):
                    print(f"Cannot find {self.camID} at {self.camLocation} - {increment}")
                    increment += 1
                    sleep(1)
            
                self.print_preamble()
                cam = self.get_camera(self.camID)
                with Vimba.get_instance() as vimba:
                    vimba.register_camera_change_handler(camera_change_handler)
                    with cam:
                        try:
                            self.setup_camera(cam)
                            self.start(cam)
                            capturing = True
                            print(f"{self.camID} at {self.camLocation} in position {self.camPosition} is connected to the server.")
                            while capturing:
                                sleep(1)
                                continue
                            cam.stop_streaming()
                        except Exception as e:
                            print(f"Exception has occurred: {e}")
                print(f"{self.camID} at {self.camLocation} in position {self.camPosition} is disconnected from the server.")
                
        except KeyboardInterrupt:
            print("Camera streaming has stopped")

    def print_preamble(self):
        print("////////////////////////////////////////////////////////")
        print("    /// Vimba API Asynchronous Grab with OpenCV ///")
        print("////////////////////////////////////////////////////////\n")

    def print_usage(self):
        print("Usage:")
        print("    python asynchronous_grab_opencv.py [camera_id]")
        print("    python asynchronous_grab_opencv.py [/h] [-h]")
        print()
        print("Parameters:")
        print("    camera_id   ID of the camera to use (using first camera if not specified)")
        print()

    def abort(self, reason: str, return_code: int = 1, usage: bool = False):
        print(reason + "\n")
        if usage:
            self.print_usage()
        sys.exit(return_code)

    def check_camera_exists(self, camera_id: str) -> bool:
        with Vimba.get_instance() as vimba:
            try:
                vimba.get_camera_by_id(camera_id)
                return True
            except Exception as e:
                print(e)
                return False

    def get_camera(self, camera_id: Optional[str]) -> Camera:
        with Vimba.get_instance() as vimba:
            if camera_id:
                try:
                    print("Camera ID is " + camera_id)
                    return vimba.get_camera_by_id(camera_id)

                except VimbaCameraError:
                    self.abort("Failed to access Camera '{}'. Abort.".format(camera_id))
            else:
                cams = vimba.get_all_cameras()
                if not cams:
                    self.abort("No Cameras accessible. Abort.")
                return cams[0]

    def setup_camera(self, cam: Camera) -> None:
        try:
            cam.GVSPAdjustPacketSize.run()
            while not cam.GVSPAdjustPacketSize.is_done():
                pass
        except (AttributeError, VimbaFeatureError):
            pass
        cam.set_pixel_format(PixelFormat.BayerRG8)

        # Sample code below updates the user set id for the camera - hardcoded to 1 currently
        print("setting user set id")
        cam.get_feature_by_name("UserSetSelector").set(1)
        cmd = cam.get_feature_by_name("UserSetLoad")
        cmd.run()
        while not cmd.is_done():
            pass

    def __enter__(self):
        """Method to be called upon entrance to a context with a context manager.

        Returns:
            CameraCapture: The initialized `CameraCapture` instance.
        """
        return self

    def start(self, cam: Camera):
        # Start the stream
        cam.start_streaming(handler=self.frame_handler, buffer_count=10)

    def frame_handler(self, cam: Camera, frame: Frame):
       
        src_frame = frame
        if frame.get_status() == FrameStatus.Complete:
            # Increment counts
            self.frameCount += 1
            self.frameRateCount += 1
            print('{} acquired {}'.format(cam, frame), flush=True)
            print(f"[{datetime.now()}] Received frame {frame} for cap camera.")
            frame = np.frombuffer(frame._buffer, dtype=np.uint8).reshape(frame._frame.height, frame._frame.width)
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2RGB)
            # frame_optimized = frame_resize(frame, self.targetDim)

            if self.camTrigger:
                pass
            else:
                if self.inferenceFPS > 0:
                    if self.frameRateCount == int(self.camFPS/self.inferenceFPS): 
                        self.frameRateCount = 0 
                        pass   

            self.cycle_begin = time.time()
            
            if ((self.modelAcvOcr == True) and (self.modelAcvOcrSecondary != True)):
                model_type = 'OCR'
                frame_optimized = frame_resize(frame, self.targetDim, model = "ocr")
                headers = {'Content-Type': 'application/octet-stream'}
                encodedFrame = cv2.imencode('.jpg', frame_optimized)[1].tobytes()
                try:
                    ocr_response = requests.post(self.modelAcvOcrUri, headers = headers, data = encodedFrame)
                    ocr_url = ocr_response.headers["Operation-Location"]
                    result = None
                    while result is None:
                        result = self.get_response(ocr_url)
                except Exception as e:
                    print('Send to OCR Exception -' + str(e))
                    result = "[]"

            elif self.modelAcvOD:
                model_type = 'Object Detection'
                frame_optimized = frame_resize(frame, self.targetDim, model = "acv")
                from inference.ort_acv_predict import predict_acv
                pil_frame = Image.fromarray(frame_optimized)
                result = predict_acv(pil_frame)
            elif self.modelYolov5:
                model_type = 'Object Detection'
                frame_optimized = frame_resize(frame, self.targetDim, model = "yolov5")
                from inference.ort_yolov5 import predict_yolov5
                result = predict_yolov5(frame_optimized)
            elif self.modelFasterRCNN:
                model_type = 'Object Detection'
                frame_optimized = frame_resize(frame, self.targetDim, model = "faster_rcnn")
                from inference.ort_faster_rcnn import predict_faster_rcnn
                result = predict_faster_rcnn(frame_optimized)
            elif self.modelMaskRCNN:
                model_type = 'Instance Segmentation'
                frame_optimized = frame_resize(frame, self.targetDim, model = "mask_rcnn")
                from inference.ort_mask_rcnn import predict_mask_rcnn
                result = predict_mask_rcnn(frame_optimized)
            elif self.modelClassMultiLabel:
                model_type = 'Multi-Label Classification'
                frame_optimized = frame_resize(frame, self.targetDim, model = "classification")
                from inference.ort_class_multi_label import predict_class_multi_label
                result = predict_class_multi_label(frame_optimized)
            elif self.modelClassMultiClass:
                model_type = 'Multi-Class Classification'
                frame_optimized = frame_resize(frame, self.targetDim, model = "classification")
                from inference.ort_class_multi_class import predict_class_multi_class
                result = predict_class_multi_class(frame_optimized)
            else:
                print("No model selected")
                result = None
            
            if result is not None:
                print(json.dumps(result))


            now = datetime.now()
            created = now.isoformat()
            unique_id = str(uuid.uuid4())
            filetime = now.strftime("%Y%d%m%H%M%S%f")
            annotatedName = f"{self.camLocation}-{self.camPosition}-{filetime}-annotated.jpg"
            annotatedPath = os.path.join('/images_volume', annotatedName)
            frameFileName = f"{self.camLocation}-{self.camPosition}-{filetime}-rawframe.jpg"
            frameFilePath = os.path.join('/images_volume', frameFileName)
            retrainFileName = f"{self.camLocation}-{self.camPosition}-{filetime}-retrain.jpg"
            retrainFilePath = os.path.join('/images_volume', retrainFileName)
            
            if result['predictions'] == "[]":
                detection_count = 0
            else:
                detection_count = len(result['predictions'])
            t_infer = result["inference_time"]
            print(f"Detection Count: {detection_count}")

            if ((model_type == 'OCR') and (self.modelAcvOcrSecondary == False)):

                            print(f'[{datetime.now()}] Results: {result["analyzeResult"]["readResults"]}')

                            # Add additional logic to extract desired text from OCR if needed and/or annotate frame with
                            # the bounding box of the text scene.

                            ocr_inference_obj = {
                                'model_name': self.model_name,
                                'object_detected': obj_det_val,
                                'camera_id': self.camID,
                                'camera_name': f"{self.camLocation}-{self.camPosition}",
                                'raw_image_name': frameFileName,
                                'raw_image_local_path': frameFilePath,
                                'annotated_image_name': frameFileName,
                                'annotated_image_path': frameFilePath,
                                'inferencing_time': t_infer,
                                'created': created,
                                'unique_id': unique_id,
                                'detected_objects': result["analyzeResult"]["readResults"]
                            }

                            sql_insert = InsertInference(Allied_GVSP_Camera.sql_state, detection_count, inference_obj)
                            Allied_GVSP_Camera.sql_state = sql_insert                      
                            self.send_to_upstream(json.dumps(ocr_inference_obj))

            elif model_type == 'Object Detection':
                detection_count = len(result['predictions'])
                t_infer = result["inference_time"]
                print(f"Detection Count: {detection_count}")
                if detection_count > 0:
                    obj_det_val = 1
                    annotated_frame = frame_optimized.copy()
                    for i in range(detection_count):
                        tag_name = result['predictions'][i]['labelName']
                        probability = round(result['predictions'][i]['probability'],2)
                        bounding_box = result['predictions'][i]['bbox']
                        image_text = f"{tag_name}@{probability}%"
                        color = (0, 255, 0)
                        thickness = 1
                        if bounding_box:
                            if self.modelAcvOD:
                                height, width, channel = annotated_frame.shape
                                xmin = int(bounding_box["left"] * width)
                                xmax = int((bounding_box["left"] * width) + (bounding_box["width"] * width))
                                ymin = int(bounding_box["top"] * height)
                                ymax = int((bounding_box["top"] * height) + (bounding_box["height"] * height))
                            else:
                                xmin = int(bounding_box["left"])
                                xmax = int(bounding_box["width"])
                                ymin = int(bounding_box["top"])
                                ymax = int(bounding_box["height"])
                            start_point = (int(bounding_box["left"]), int(bounding_box["top"]))
                            end_point = (int(bounding_box["width"]), int(bounding_box["height"]))
                            annotated_frame = cv2.rectangle(annotated_frame, start_point, end_point, color, thickness)
                            annotated_frame = cv2.putText(annotated_frame, image_text, start_point, fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = .6, color = (255,0, 0))
                        if self.modelAcvOcrSecondary:
                            xmin = xmin - 5
                            xmax - xmax + 5
                            ymin = ymin - 5
                            ymax = ymax + 5
                            ocrFrame = frame_optimized[ymin:ymax, xmin:xmax]
                            ocrFrameName = f"{tag_name}-{i}-{self.camLocation}-{self.camPosition}-{filetime}.jpg"
                            ocrFramePath = os.path.join('/images_volume', ocrFrameName) 
                                
                            ocrGreyFrame = cv2.cvtColor(ocrFrame, cv2.COLOR_BGR2GRAY)

                            model_type = 'OCR'
                            headers = {'Content-Type': 'application/octet-stream'}
                            encodedFrame = cv2.imencode('.jpg', ocrGreyFrame)[1].tobytes()
                            try:
                                ocr_response = requests.post(self.modelAcvOcrUri, headers = headers, data = encodedFrame)
                                ocr_url = ocr_response.headers["Operation-Location"]
                                result = None
                                while result is None:
                                    result = self.get_response(ocr_url)
                            except Exception as e:
                                print('Send to OCR Exception -' + str(e))
                                result = "[]"
                            
                            print(f'[{datetime.now()}] Results: {result["analyzeResult"]["readResults"]}')





                            FrameSave(ocrFramePath, ocrGreyFrame)

                    FrameSave(annotatedPath, annotated_frame)
                    annotated_msg = {
                    'fs_name': "images-annotated",
                    'img_name': annotatedName,
                    'location': self.camLocation,
                    'position': self.camPosition,
                    'path': annotatedPath
                    }
                    self.send_to_upload(json.dumps(annotated_msg))  

                else:
                    if self.storeAllInferences:
                        obj_det_val = 0
                        annotatedName = frameFileName
                        annotatedPath = frameFilePath

                inference_obj = {
                    'model_name': self.model_name,
                    'object_detected': obj_det_val,
                    'camera_id': self.camID,
                    'camera_name': f"{self.camLocation}-{self.camPosition}",
                    'raw_image_name': frameFileName,
                    'raw_image_local_path': frameFilePath,
                    'annotated_image_name': annotatedName,
                    'annotated_image_path': annotatedPath,
                    'inferencing_time': t_infer,
                    'created': created,
                    'unique_id': unique_id,
                    'detected_objects': result['predictions']
                    }

                sql_insert = InsertInference(Allied_GVSP_Camera.sql_state, detection_count, inference_obj)
                Allied_GVSP_Camera.sql_state = sql_insert                      
                self.send_to_upstream(json.dumps(inference_obj))   

            elif model_type == 'Instance Segmentation':
                detection_count = len(result['predictions'])
                t_infer = result["inference_time"]
                annotatedName = result["annotated_image_name"]
                annotatedPath = result["annotated_image_path"] 
                print(f"Detection Count: {detection_count}")
                if detection_count > 0:
                    obj_det_val = 1

                #   Frame upload
                    annotated_msg = {
                    'fs_name': "images-annotated",
                    'img_name': annotatedName,
                    'location': self.camLocation,
                    'position': self.camPosition,
                    'path': annotatedPath
                    }
                    self.send_to_upload(json.dumps(annotated_msg))  

                else:
                    if self.storeAllInferences:
                        obj_det_val = 0
                        annotatedName = frameFileName
                        annotatedPath = frameFilePath

                inference_obj = {
                    'model_name': self.model_name,
                    'object_detected': obj_det_val,
                    'camera_id': self.camID,
                    'camera_name': f"{self.camLocation}-{self.camPosition}",
                    'raw_image_name': frameFileName,
                    'raw_image_local_path': frameFilePath,
                    'annotated_image_name': annotatedName,
                    'annotated_image_path': annotatedPath,
                    'inferencing_time': t_infer,
                    'created': created,
                    'unique_id': unique_id,
                    'detected_objects': result['predictions']
                    }

                sql_insert = InsertInference(Allied_GVSP_Camera.sql_state, detection_count, inference_obj)
                Allied_GVSP_Camera.sql_state = sql_insert                      
                self.send_to_upstream(json.dumps(inference_obj))                
            
            elif model_type == 'Multi-Label Classification' or model_type == 'Multi-Label Classification':
                detection_count = len(result['predictions'])
                t_infer = result["inference_time"]
                annotatedName = result["annotated_image_name"]
                annotatedPath = result["annotated_image_path"] 
                print(f"Detection Count: {detection_count}")
                if detection_count > 0:
                    obj_det_val = 1

                else:
                    if self.storeAllInferences:
                        obj_det_val = 0
                        annotatedName = frameFileName
                        annotatedPath = frameFilePath

                inference_obj = {
                    'model_name': self.model_name,
                    'object_detected': obj_det_val,
                    'camera_id': self.camID,
                    'camera_name': f"{self.camLocation}-{self.camPosition}",
                    'raw_image_name': frameFileName,
                    'raw_image_local_path': frameFilePath,
                    'annotated_image_name': frameFileName,
                    'annotated_image_path': frameFilePath,
                    'inferencing_time': t_infer,
                    'created': created,
                    'unique_id': unique_id,
                    'detected_objects': result['predictions']
                    }

                sql_insert = InsertInference(Allied_GVSP_Camera.sql_state, detection_count, inference_obj)
                Allied_GVSP_Camera.sql_state = sql_insert                      
                self.send_to_upstream(json.dumps(inference_obj))


            print(f"Frame count = {self.frameCount}")
            self.cycle_end = time.time()
            self.t_full_cycle = (self.cycle_end - self.cycle_begin)*1000
            print("Cycle Time in ms: {}".format(self.t_full_cycle))
            
            self.frameRateCount = 0
            FrameSave(frameFilePath, frame_optimized)

            if (self.storeRawFrames == True):
                frame_msg = {
                'fs_name': "images-frame",
                'img_name': frameFileName,
                'location': self.camLocation,
                'position': self.camPosition,
                'path': frameFilePath
                }
                self.send_to_upload(json.dumps(frame_msg))

            if (self.frameCount*(self.inferenceFPS/self.camFPS)) % self.retrainInterval == 0:
                FrameSave(retrainFilePath, frame)
                retrain_msg = {
                'fs_name': "images-retraining",
                'img_name': retrainFileName,
                'location': self.camLocation,
                'position': self.camPosition,
                'path': retrainFilePath
                }
                self.send_to_upload(json.dumps(retrain_msg))


        cam.queue_frame(src_frame)