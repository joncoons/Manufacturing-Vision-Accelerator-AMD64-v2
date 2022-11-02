import json
import numpy as np
import onnxruntime as ort
import time
import os
from datetime import datetime
import tempfile

providers = [
    'CUDAExecutionProvider',
    'CPUExecutionProvider',
]
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

class ONNXRuntimeObjectDetection():

    def __init__(self, model_path, classes, target_dim, target_prob, target_iou):
        self.target_dim = target_dim
        self.target_prob = target_prob
        self.target_iou = target_iou
        
        self.device_type = ort.get_device()
        print(f"ORT device: {self.device_type}")

        # with tempfile.TemporaryDirectory() as model_store:
        #     model_opt_path = os.path.join(model_store, os.path.basename(model_path))
        #     sess_options = ort.SessionOptions()
        #     sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        #     sess_options.optimized_model_filepath = model_opt_path
        #     self.session = ort.InferenceSession(model_path, sess_options, providers=providers)

        self.session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        batch, channel, height_onnx, width_onnx = self.session.get_inputs()[0].shape
        self.batch = batch
        self.channel = channel
        self.height_onnx = height_onnx
        self.width_onnx = width_onnx

        self.classes = classes
        self.num_classes = len(classes)
             
    def predict(self, pre_image):
        sess_input = self.session.get_inputs()
        sess_output = self.session.get_outputs()
        
        output_names = [output.name for output in sess_output]
        outputs = self.session.run(output_names=output_names, input_feed={sess_input[0].name:pre_image})
        
        def _get_box_dims(image_shape, box):
            box_keys = ['left', 'top', 'width', 'height']

            bbox = dict(zip(box_keys, [coordinate.item() for coordinate in box]))

            return bbox

        def _get_prediction(boxes, labels, scores, image_shape, classes):
            raw_pred = []
            for box, label_index, score in zip(boxes, scores, labels):
                box_dims = _get_box_dims(image_shape, box)

                prediction = {  
                    'probability': score.item(),
                    'labelId': label_index.item(),
                    'labelName': classes[label_index],
                    'bbox': box_dims
                }
                raw_pred.append(prediction)

            return raw_pred

        boxes, labels, scores = outputs[0], outputs[1], outputs[2]
        unfiltered_pred = _get_prediction(boxes, labels, scores, (self.height_onnx, self.width_onnx), self.classes)
        filtered_pred = [x for x in unfiltered_pred if x['probability'] >= self.target_prob]

        if len(filtered_pred) > 0:
            print(json.dumps(filtered_pred, indent=1))
            return filtered_pred
        else:
            print("No predictions passed the threshold")  
            return []

def log_msg(msg):
    print("{}: {}".format(datetime.now(), msg))

def checkModelExtension(fp):
  ext = os.path.splitext(fp)[-1].lower()
  if(ext != ".onnx"):
    raise Exception(fp, "is an unknown file format. Use the model ending with .onnx format")
  if not os.path.exists(fp):
    raise Exception("[ ERROR ] Path of the onnx model file is Invalid")

def initialize_retinanet(model_path, labels_path, target_dim, target_prob, target_iou):
    print('Loading classes...\n', end='')
    checkModelExtension(model_path)
    with open(labels_path) as f:
        classes = json.load(f) 
    print('Loading model...\n', end='')
    global ort_model
    ort_model = ONNXRuntimeObjectDetection(model_path, classes, target_dim, target_prob, target_iou)
    print('Success!')

def predict_retinanet(image):
    log_msg('Predicting image')
    frame = image.transpose(2,0,1)
    mean_vec = np.array([0.485, 0.456, 0.406])
    std_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(frame.shape).astype('float32')
    for i in range(frame.shape[0]):
        norm_img_data[i,:,:] = (frame[i,:,:] / 255 - mean_vec[i]) / std_vec[i]
    frame = np.expand_dims(norm_img_data, axis=0)
    
    t1 = time.time()
    predictions = ort_model.predict(frame)
    t2 = time.time()
    t_infer = (t2-t1)*1000
    response = {
        'created': datetime.utcnow().isoformat(),
        'inference_time': t_infer,
        'predictions': predictions
        }
    return response

def warmup_image(batch_size, warmup_dim):
    for _ in range(batch_size):
        yield np.zeros([warmup_dim, warmup_dim, 3], dtype=np.uint8)

