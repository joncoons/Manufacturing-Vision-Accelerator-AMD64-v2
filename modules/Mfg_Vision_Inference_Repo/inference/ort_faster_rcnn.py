import json
import numpy as np
import onnxruntime as ort
import time
import os
from datetime import datetime

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

# provider = [
#     'CPUExecutionProvider',
# ]

class ONNXRuntimeObjectDetection():

    def __init__(self, model_path, classes, target_dim, target_prob, target_iou):
        self.target_dim = target_dim
        self.target_prob = target_prob
        self.target_iou = target_iou
        
        self.device_type = ort.get_device()
        print(f"ORT device: {self.device_type}")

        self.session = ort.InferenceSession(model_path, providers=providers)

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

        self.input_name = self.session.get_inputs()[0].name
        batch, channel, height_onnx, width_onnx = self.session.get_inputs()[0].shape
        self.batch = batch
        self.channel = channel
        self.height_onnx = height_onnx
        self.width_onnx = width_onnx
        # self.sess_input = self.session.get_inputs()
        # self.sess_output = self.session.get_outputs()
        # self.output_name = self.session.get_outputs()[0].name
        # self.batch_size = self.session.get_inputs()[0].shape[0]
        # self.channels = self.session.get_inputs()[0].shape[1]
        # self.img_size_h = self.session.get_inputs()[0].shape[2]
        # self.img_size_w = self.session.get_inputs()[0].shape[3]
        # print("Input: {} Output: {} Batch Size: {} Model ImgH: {} Model ImgW: {}".format(self.input_name,self.output_name,self.batch_size,self.img_size_h,self.img_size_w))
        # self.is_fp16 = self.session.get_inputs()[0].type == 'tensor(float16)'
        self.classes = classes
        self.num_classes = len(classes)
             
    def predict(self, pp_image, image):
        inputs = pp_image
        img_shape_batch = pp_image.shape[0]
        # print(f"img_batch: {img_shape_batch}")
        # if self.is_fp16:
        #     inputs = inputs.astype(np.float16)
        output_names = [output.name for output in self.sess_output]
        outputs = self.session.run(output_names=output_names, input_feed={self.sess_input[0].name:inputs})
        # img = image
        # batch_num = int(1)

        ONNXRuntimeObjectDetection.inference_count = 0
        
        def _get_box_dims(image_shape, box):
            box_keys = ['left', 'top', 'width', 'height']
            height, width = image_shape[0], image_shape[1]

            bbox = dict(zip(box_keys, [coordinate.item() for coordinate in box]))

            # bbox['x1'] = bbox['x1'] * 1.0 / width
            # bbox['x2'] = bbox['x2'] * 1.0 / width
            # bbox['y1'] = bbox['y1'] * 1.0 / height
            # bbox['y2'] = bbox['y2'] * 1.0 / height

            return bbox

        def _get_prediction(boxes, labels, scores, image_shape, classes):
            raw_pred = []
            for box, label_index, score in zip(boxes, labels, scores):
                box_dims = _get_box_dims(image_shape, box)

                # box_record = {'box': box_dims,
                #       'label': classes[label_index],
                #       'score': score.item()}

                prediction = {  
                    'probability': score.item(),
                    'labelId': label_index.item(),
                    'labelName': classes[label_index],
                    'bbox': box_dims
                }
                raw_pred.append(prediction)

            return raw_pred

        for img_batch in range(0, img_shape_batch*3, 3):
        # in case of retinanet change the order of boxes, labels, scores to boxes, scores, labels
        # confirm the same from order of boxes, labels, scores output_names 
            boxes, labels, scores = outputs[img_batch], outputs[img_batch + 1], outputs[img_batch + 2]
            unfiltered_pred = _get_prediction(boxes, labels, scores, (self.height_onnx, self.width_onnx), self.classes)
            print(f"Unfiltered predictions: {unfiltered_pred}")
            filtered_pred = [x for x in unfiltered_pred if x['probability'] >= self.target_prob]
            print(f"Filtered predictions: {filtered_pred}")
            # for prediction in filtered_pred:

            if len(filtered_pred) > 0:
                print(f"Filtered predictions: {filtered_pred}")
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

def initialize_faster_rcnn(model_path, labels_path, target_dim, target_prob, target_iou):
    print('Loading classes...\n', end='')
    checkModelExtension(model_path)
    with open(labels_path) as f:
        classes = json.load(f) 
    print('Loading model...\n', end='')
    global ort_model
    ort_model = ONNXRuntimeObjectDetection(model_path, classes, target_dim, target_prob, target_iou)
    print('Success!')

def predict_faster_rcnn(image):
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
    
    # frame /= 255.0 # normalize pixels
    # print(f"Batch-Size, Channel, Height, Width : {frame.shape}")
    t1 = time.time()
    predictions = ort_model.predict(frame, image)
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

