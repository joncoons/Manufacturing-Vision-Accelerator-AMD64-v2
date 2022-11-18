import os
from datetime import datetime
from urllib.request import urlopen
import time
import io

import onnxruntime
import onnx

import numpy as np
from PIL import Image, ImageDraw

providers = [
    'CUDAExecutionProvider',
    'CPUExecutionProvider',
]

class ONNXRuntimeACVClass:
    """Object Detection class for ONNX Runtime"""
    def __init__(self, model_filename, labels):
        self.session = onnxruntime.InferenceSession(str(model_filename), providers=providers)
        assert len(self.session.get_inputs()) == 1
        self.input_shape = self.session.get_inputs()[0].shape[2:]
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}[self.session.get_inputs()[0].type]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.target_prob = float(os.environ["PROB_THRES"])
        self.is_bgr = False
        self.is_range255 = False
        onnx_model = onnx.load(model_filename)
        for metadata in onnx_model.metadata_props:
            if metadata.key == 'Image.BitmapPixelFormat' and metadata.value == 'Bgr8':
                self.is_bgr = True
            elif metadata.key == 'Image.NominalPixelRange' and metadata.value == 'NominalRange_0_255':
                self.is_range255 = True
        
        self.classes = labels
        self.num_classes = len(self.classes)
        
    def predict(self, image_array):
        outputs = self.session.run(self.output_names, {self.input_name: image_array.astype(self.input_type)})
        # print(f"Predictions all : {outputs}")
        
        scores = outputs[0]
        # print(f"Scores all : {scores}")
        
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / np.sum(e_x, axis=1, keepdims=True)

        pred_list = []
        conf_scores = softmax(scores)
        # print(f"Confidence scores: {conf_scores}")
        score_max = np.max(conf_scores, axis=1)
        # print(f"Score Max: {score_max[0]}")
        if score_max[0] > self.target_prob:
            class_pred = np.argmax(conf_scores, axis=1)
            # print("predicted classes:", ([(class_idx, self.classes[class_idx]) for class_idx in class_pred]))
            # to match SQL table schema
            x1 = int(0)
            y1 = int(0)
            x2 = int(0)
            y2 = int(0)
            for class_idx in class_pred:
                pred_list.append({
                    "probability": conf_scores[0][class_idx].item(),
                    "labelId": class_idx.item(),
                    "labelName": self.classes[class_idx],
                    "bbox": {
                        'left': x1,
                        'top': y1,
                        'width': x2,
                        'height': y2
                    }
                })
            print(f"Predictions : {pred_list}")

        else:
            print('No prediction above threshold')

        return pred_list

def log_msg(msg):
    print("{}: {}".format(datetime.now(), msg))

def initialize_acv_mc_class(modelPath, labelPath):
    """Load labels and onnx model"""
    print('Loading labels...', end='')
    with open(labelPath, 'r') as f:
        labels = [l.strip() for l in f.readlines()]
    print("{} found. Success!".format(len(labels)))
    
    print('Loading model...', end='')
    global od_model
    od_model = ONNXRuntimeACVClass(modelPath, labels)
    print('Success!')

def predict_acv_mc_class(image):
    log_msg('Predicting image')
    frame = image.transpose(2, 0, 1)
    mean_vec = np.array([0.485, 0.456, 0.406])
    std_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(frame.shape).astype('float32')
    print(f"Frame shape: {frame.shape}")
    for i in range(frame.shape[0]):
        norm_img_data[i,:,:] = (frame[i,:,:] / 255 - mean_vec[i]) / std_vec[i]
    frame = np.expand_dims(norm_img_data, axis=0)
    frame = frame[:, (2, 1, 0), :, :] # BGR to RGB

    # print(f"Batch-Size, Channel, Height, Width : {frame.shape}")

    t1 = time.time()
    img_predict = od_model.predict(frame)
    t2 = time.time()
    t_infer = round((t2-t1)*1000,2)
    response = {
        'created': datetime.utcnow().isoformat(),
        'inference_time': t_infer,
        'predictions': img_predict
        }
    print(f"Response : {response}")
    return response


