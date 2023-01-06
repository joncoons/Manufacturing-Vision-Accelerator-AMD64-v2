import json
import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import tempfile
from datetime import datetime
from capture.frame_save import FrameSave

providers = [
    'CUDAExecutionProvider',
    'CPUExecutionProvider',
]

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

        self.session = ort.InferenceSession(model_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        batch, channel, height_onnx, width_onnx = self.session.get_inputs()[0].shape
        print(f'batch, channel, height_onnx, width_onnx : {batch}, {channel}, {height_onnx}, {width_onnx}')
        self.batch = batch
        self.channel = channel
        self.height_onnx = height_onnx
        self.width_onnx = width_onnx

        self.classes = classes
        self.num_classes = len(classes)
             
    def predict(self, normalized_image, image):
        sess_input = self.session.get_inputs()
        sess_output = self.session.get_outputs()
        output_names = [output.name for output in sess_output]
        outputs = self.session.run(output_names=output_names, input_feed={sess_input[0].name:normalized_image})
        
        def _get_box_dims(image_shape, box):
            box_keys = ['xmin', 'ymin', 'xmax', 'ymax']
            height, width = image_shape[0], image_shape[1]
            print(f"height: {height}, width: {width}")
            bbox = dict(zip(box_keys, [int(coordinate.item()) for coordinate in box]))
            return bbox

        def _get_prediction(boxes, labels, scores, masks, image_shape, classes):
            raw_pred = []
            
            color = (0, 255, 0)
            thickness = 1
            for mask, box, label_index, score in zip(masks, boxes, labels, scores):
                if score <= self.target_prob:
                    continue
                bbox = _get_box_dims(image_shape, box)
                probability = round(score.item(),2)
                labelId = label_index.item()
                labelName = classes[label_index]
                # image_text = f"{labelName}@{probability*100}%"
                image_text = f"{probability*100}%"
                mask = mask[0, :, :, None]
                print(f'mask.shape: {mask.shape}')
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 0, 0, interpolation = cv2.INTER_NEAREST)    
                mask = mask > self.target_prob
                print(f"mask: {mask}")
                image_masked = image.copy()
                image_masked[mask] = (0, 255, 100)
                alpha = 0.3  # alpha blending with range 0 to 1
                annotated_frame = cv2.addWeighted(image_masked, alpha, image, 1 - alpha,0, image)
                # start_point = (int(bbox['xmin'])), int(bbox['ymin'])
                # end_point = (int(bbox['xmax']), int(bbox['ymax']))
                # annotated_frame = cv2.rectangle(annotated_frame, start_point, end_point, color, thickness)
                # annotated_frame = cv2.putText(annotated_frame, image_text, start_point, fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = .5, color = (255, 255, 255))

                # FrameSave(annotated_mask_path, annotated_frame)           

                prediction = {  
                    'probability': probability,
                    'labelId': labelId,
                    'labelName': labelName,
                    'bbox': bbox,
                }
                raw_pred.append(prediction)

            return raw_pred, annotated_frame

        boxes, labels, scores, masks = outputs[0], outputs[1], outputs[2], outputs[3]
        predictions, masked_image = _get_prediction(boxes, labels, scores, masks, (self.height_onnx, self.width_onnx), self.classes)

        if len(predictions) > 0:
            print(f"Filtered predictions: {predictions}")
            return predictions, masked_image
        else:
            print("No predictions passed the threshold")  
            return [], None

def log_msg(msg):
    print("{}: {}".format(datetime.now(), msg))

def checkModelExtension(fp):
  ext = os.path.splitext(fp)[-1].lower()
  if(ext != ".onnx"):
    raise Exception(fp, "is an unknown file format. Use the model ending with .onnx format")
  if not os.path.exists(fp):
    raise Exception("[ ERROR ] Path of the onnx model file is Invalid")

def initialize_mask_rcnn(model_path, labels_path, target_dim, target_prob, target_iou):
    print('Loading classes...\n', end='')
    checkModelExtension(model_path)
    with open(labels_path) as f:
        classes = json.load(f) 
    print('Loading model...\n', end='')
    global ort_model
    ort_model = ONNXRuntimeObjectDetection(model_path, classes, target_dim, target_prob, target_iou)
    print('Success!')

def predict_mask_rcnn(image):
    log_msg('Predicting image')
    frame = image.transpose(2,0,1)
    mean_vec = np.array([0.485, 0.456, 0.406])
    std_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(frame.shape).astype('float32')
    for i in range(frame.shape[0]):
        norm_img_data[i,:,:] = (frame[i,:,:] / 255 - mean_vec[i]) / std_vec[i]
    frame = np.expand_dims(norm_img_data, axis=0)

    t1 = time.time()
    predictions, masked_image = ort_model.predict(frame, image)
    t2 = time.time()
    t_infer = (t2-t1)*1000
    response = {
        'created': datetime.utcnow().isoformat(),
        'inference_time': t_infer,
        'predictions': predictions
        }
    return response, masked_image
# 
def warmup_image(batch_size, warmup_dim):
    for _ in range(batch_size):
        yield np.zeros([warmup_dim, warmup_dim, 3], dtype=np.uint8)

