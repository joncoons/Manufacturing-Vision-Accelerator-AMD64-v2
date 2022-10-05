import json
# import torch
import numpy as np
from PIL import Image
import onnxruntime as ort
import time
import os
from datetime import datetime
# from inference.utils.general import non_max_suppression

# providers = [
#     ('CUDAExecutionProvider', {
#         'device_id': 0,
#         'arena_extend_strategy': 'kSameAsRequested ',
#         'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
#         'cudnn_conv_algo_search': 'DEFAULT',
#         'do_copy_in_default_stream': True,
#     }),
#     'CPUExecutionProvider',
# ]
providers = [
    'CUDAExecutionProvider',
    'CPUExecutionProvider',
]

class ONNXRuntimeClassificationMultiClass():

    def __init__(self, model_path, classes, target_dim, target_prob, target_iou):
        self.target_dim = target_dim
        self.target_prob = target_prob
        self.target_iou = target_iou
        
        self.device_type = ort.get_device()
        print(f"ORT device: {self.device_type}")

        self.session = ort.InferenceSession(model_path, providers=providers)
        batch, channel, height_onnx_crop_size, width_onnx_crop_size = self.session.get_inputs()[0].shape
        batch, channel, height_onnx_crop_size, width_onnx_crop_size
        self.batch = batch
        self.channel = channel
        self.height_onnx = height_onnx_crop_size
        self.width_onnx = width_onnx_crop_size
        self.sess_input = self.session.get_inputs()
        self.sess_output = self.session.get_outputs()
        print(f"No. of inputs : {len(self.sess_input)}, No. of outputs : {len(self.sess_output)}") 
        for idx, input_ in enumerate(range(len(self.sess_input))):
            input_name = self.sess_input[input_].name
            input_shape = self.sess_input[input_].shape
            input_type = self.sess_input[input_].type
            print(f"{idx} Input name : { input_name }, Input shape : {input_shape}, \
            Input type  : {input_type}")  

        for idx, output in enumerate(range(len(self.sess_output))):
            output_name = self.sess_output[output].name
            output_shape = self.sess_output[output].shape
            output_type = self.sess_output[output].type
            print(f" {idx} Output name : {output_name}, Output shape : {output_shape}, \
            Output type  : {output_type}") 

        self.classes = classes
        self.num_classes = len(classes)
             
    def predict(self, pp_image, image):
        inputs = pp_image
        # if self.is_fp16:
        #     inputs = inputs.astype(np.float16)
        output_names = [output.name for output in self.sess_output]
        print(output_names)
        outputs = self.session.run(output_names=output_names, input_feed={self.sess_input[0].name: inputs})
        print(f"Predictions all : {outputs}")
        
        scores = outputs[0]
        print(f"Scores all : {scores}")

        # ////////////// Torch /////////////////
        # conf_scores = torch.nn.functional.softmax(torch.from_numpy(scores), dim=1)
        # print(f"Conf scores : {conf_scores}")
        # class_preds = torch.argmax(conf_scores, dim=1)
        # print(f"Class preds : {class_preds}")
        # print("predicted classes:", ([(class_idx.item(), self.classes[class_idx]) for class_idx in class_preds]))
        # predictions = []
        # for class_idx in class_preds:
        #     predictions.append({
        #         "probability": conf_scores[0][class_idx].item(),
        #         "labelId": class_idx.item(),
        #         "labelName": self.classes[class_idx],
        #         "bbox": [0, 0, 0, 0]
        #     })

        # print(f"Predictions : {predictions}")

        # ////////////// No Torch /////////////////
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / np.sum(e_x, axis=1, keepdims=True)

        conf_scores = softmax(scores)
        print(f"Confidence scores: {conf_scores}")
        class_preds = np.argmax(conf_scores, axis=1)
        # print(f"Class predictions: {class_preds}")
        # print(f"Classes: {self.classes}")
        print("predicted classes:", ([(class_idx, self.classes[class_idx]) for class_idx in class_preds]))
        pred_list = []
        # to match SQL table schema
        x1 = int(0)
        y1 = int(0)
        x2 = int(0)
        y2 = int(0)
        for class_idx in class_preds:
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

        return pred_list

def log_msg(msg):
    print("{}: {}".format(datetime.now(), msg))

def checkModelExtension(fp):
  ext = os.path.splitext(fp)[-1].lower()
  if(ext != ".onnx"):
    raise Exception(fp, "is an unknown file format. Use the model ending with .onnx format")
  if not os.path.exists(fp):
    raise Exception("[ ERROR ] Path of the onnx model file is Invalid")

def initialize_class_multi_class(model_path, labels_path, target_dim, target_prob, target_iou):
    print('Loading classes...\n', end='')
    checkModelExtension(model_path)
    with open(labels_path) as f:
        classes = json.load(f) 
    print('Loading model...\n', end='')
    global ort_model
    ort_model = ONNXRuntimeClassificationMultiClass(model_path, classes, target_dim, target_prob, target_iou)
    print('Success!')

def predict_class_multi_class(image):
    log_msg('Predicting image')
    frame = np.asarray(image)
    frame = frame.astype(np.float32)
    frame = frame.transpose(2,0,1)
    
    mean_vec = np.array([0.485, 0.456, 0.406])
    std_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(frame.shape).astype('float32')
    for i in range(frame.shape[0]):
        norm_img_data[i,:,:] = (frame[i,:,:] / 255 - mean_vec[i]) / std_vec[i]

    frame = np.expand_dims(norm_img_data, axis=0)
    batch_size = 1
    assert batch_size == frame.shape[0]
    
    # frame /= 255.0 # normalize pixels
    print(f"Batch-Size, Channel, Height, Width : {frame.shape}")
    t1 = time.time()
    img_predict = ort_model.predict(frame, image)
    t2 = time.time()
    t_infer = round((t2-t1)*1000,2)
    response = {
        'created': datetime.utcnow().isoformat(),
        'inference_time': t_infer,
        'predictions': img_predict
        }
    print(f"Response : {response}")
    return response

def warmup_image(batch_size, warmup_dim):
    for _ in range(batch_size):
        yield np.zeros([warmup_dim, warmup_dim, 3], dtype=np.uint8)

