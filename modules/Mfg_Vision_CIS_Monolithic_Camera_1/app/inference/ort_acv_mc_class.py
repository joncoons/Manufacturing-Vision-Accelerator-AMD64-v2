# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
import pathlib
from datetime import datetime
from urllib.request import urlopen
import time
import io

import onnxruntime
import onnx

import numpy as np
from PIL import Image, ImageDraw

# MODEL_FILENAME = os.path.join('/model_volume/',os.environ["MODEL_FILE"])
# LABELS_FILENAME = os.path.join('/model_volume/',os.environ["LABEL_FILE"])

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

        self.is_bgr = False
        self.is_range255 = False
        onnx_model = onnx.load(model_filename)
        for metadata in onnx_model.metadata_props:
            if metadata.key == 'Image.BitmapPixelFormat' and metadata.value == 'Bgr8':
                self.is_bgr = True
            elif metadata.key == 'Image.NominalPixelRange' and metadata.value == 'NominalRange_0_255':
                self.is_range255 = True
        
    def predict(self, image):
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array.transpose((0, 3, 1, 2))  # => (N, C, H, W)
        if self.is_bgr:
            input_array = input_array[:, (2, 1, 0), :, :]
        if not self.is_range255:
            input_array = input_array / 255  # => Pixel values should be in range [0, 1]

        outputs = self.session.run(self.output_names, {self.input_name: input_array.astype(self.input_type)})
        pred_list = []
        x1 = int(0)
        y1 = int(0)
        x2 = int(0)
        y2 = int(0)
        # outputs[i] for i, name in enumerate(self.output_names)
        return {name: outputs[i] for i, name in enumerate(self.output_names)}

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

    w, h = image.size
    log_msg("Image size: {}x{}".format(w, h))
    t1 = time.time()
    predictions = od_model.predict(image)
    print(f'Predictions: {predictions}')
    t2 = time.time()
    t_infer = (t2-t1)*1000

    # response = predictions
    # response = {
    #             'created': datetime.utcnow().isoformat(),
    #             'inference_time': t_infer,
    #             'predictions': predictions
    #             }

    # log_msg('Results: ' + json.dumps(response))
    
    # return response

