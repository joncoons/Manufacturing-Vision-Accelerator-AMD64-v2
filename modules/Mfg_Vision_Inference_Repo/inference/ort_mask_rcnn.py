import json
import cv2
import numpy as np
import onnxruntime as ort
import time
import os
from datetime import datetime
from capture.frame_save import FrameSave

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
        print(f'batch, channel, height_onnx, width_onnx : {batch}, {channel}, {height_onnx}, {width_onnx}')
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
        image = np.array(image)
        # print(f"img_batch: {img_shape_batch}")
        # if self.is_fp16:
        #     inputs = inputs.astype(np.float16)
        output_names = [output.name for output in self.sess_output]
        print(f"Output_names: {output_names}")
        outputs = self.session.run(output_names=output_names, input_feed={self.sess_input[0].name:inputs})
        print(f"outputs: {outputs}")
        
        
        # batch_num = int(1)

        ONNXRuntimeObjectDetection.inference_count = 0
        
        def _get_box_dims(image_shape, box):
            box_keys = ['xmin', 'ymin', 'xmax', 'ymax']
            height, width = image_shape[0], image_shape[1]
            print(f"height: {height}, width: {width}")
            bbox = dict(zip(box_keys, [int(coordinate.item()) for coordinate in box]))
            return bbox

        def _get_prediction(boxes, labels, scores, masks, image_shape, classes):
            now = datetime.now()
            filetime = now.strftime("%Y%d%m%H%M%S%f")
            annotatedName = f"mask-{filetime}-annotated.jpg"
            annotatedPath = os.path.join('/images_volume', annotatedName)
            # output_counter = 0
            # output_path = "/debug/"
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
                image_text = f"{labelName}@{probability}%"

                # mask_list = mask[0, :, :, None]
                # mask_list = mask_list.tolist()

                # output_counter += 1
                # output_file = f"output_{output_counter}.txt"
                # output_full = os.path.join(output_path, output_file)
                # im_mask = mask[:, :, None]
                mask = mask[0, :, :, None]
                # mask_list=im_mask.tolist()
                # print(mask_list)
                # ONNX Azure ML guidance
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 0, 0, interpolation = cv2.INTER_NEAREST)    
                mask = mask > self.target_prob
                image_masked = image.copy()
                image_masked[mask] = (0, 255, 100)
                alpha = 0.3  # alpha blending with range 0 to 1
                annotated_frame = cv2.addWeighted(image_masked, alpha, image, 1 - alpha,0, image)
                start_point = (int(bbox['xmin'])), int(bbox['ymin'])
                end_point = (int(bbox['xmax']), int(bbox['ymax']))
                annotated_frame = cv2.rectangle(annotated_frame, start_point, end_point, color, thickness)
                annotated_frame = cv2.putText(annotated_frame, image_text, start_point, fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = .6, color = (255,0, 0))

                # # ONNX runtime github sample code
                # int_box = [int(i) for i in box]
                # mask = cv2.resize(mask, (int_box[2]-int_box[0]+1, int_box[3]-int_box[1]+1))
                # mask_filtered = mask > 0.5

                # im_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                # x_0 = max(int_box[0], 0)
                # x_1 = min(int_box[2] + 1, image.shape[1])
                # y_0 = max(int_box[1], 0)
                # y_1 = min(int_box[3] + 1, image.shape[0])
                # mask_y_0 = max(y_0 - box[1], 0)
                # mask_y_1 = mask_y_0 + y_1 - y_0
                # mask_x_0 = max(x_0 - box[0], 0)
                # mask_x_1 = mask_x_0 + x_1 - x_0
                # im_mask[y_0:y_1, x_0:x_1] = mask[
                #     mask_y_0 : mask_y_1, mask_x_0 : mask_x_1
                # ]
                # im_mask = im_mask[:, :, None]

                #  # OpenCV version 4.x
                # contours, hierarchy = cv2.findContours(
                #     im_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                # )
                # annotated_frame = cv2.drawContours(image, contours, -1, 25, 3)    

                FrameSave(annotatedPath, annotated_frame)           

                # mask_list = mask_filtered.tolist()
                # new_mask_arr = mask[mask_list]
                # new_mask_arr = new_mask_arr.tolist()
                # with open(output_full, 'w') as f:
                #     f.write(f"Mask Array: \n{new_mask_arr}\n")


                # image = np.array(image)
                # image_masked = image.copy()
                # image_masked[mask] = (0, 255, 255)
                # alpha = 0.5  # alpha blending with range 0 to 1
                # cv2.addWeighted(image_masked, alpha, image, 1 - alpha,0, image)
                # rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],\
                #                         linewidth=1, edgecolor='b', facecolor='none')

                # mask = mask[0, :, :, None]  
                # mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 0, 0, interpolation = cv2.INTER_NEAREST)
                
                # print(f"mask: {mask}")
                # box_record = {'box': box_dims,
                #       'label': classes[label_index],
                #       'score': score.item()}

                prediction = {  
                    'probability': probability,
                    'labelId': labelId,
                    'labelName': labelName,
                    'bbox': bbox,
                }
                raw_pred.append(prediction)

            return raw_pred, annotatedName, annotatedPath

        for img_batch in range(0, img_shape_batch*3, 3):
        # in case of retinanet change the order of boxes, labels, scores to boxes, scores, labels
        # confirm the same from order of boxes, labels, scores output_names 
            boxes, labels, scores, masks = outputs[img_batch], outputs[img_batch + 1], outputs[img_batch + 2], outputs[img_batch + 3]
            # boxes, labels, scores, masks = outputs[4:8]
            predictions, a_name, a_path = _get_prediction(boxes, labels, scores, masks, (self.height_onnx, self.width_onnx), self.classes)
            # print(f"Unfiltered predictions: {unfiltered_pred}")
            # for prediction in filtered_pred:

            if len(predictions) > 0:
                print(f"Filtered predictions: {predictions}")
                return predictions, a_name, a_path
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
    print(f"Batch-Size, Channel, Height, Width : {frame.shape}")
    t1 = time.time()
    predictions, a_name, a_path = ort_model.predict(frame, image)
    t2 = time.time()
    t_infer = (t2-t1)*1000
    response = {
        'created': datetime.utcnow().isoformat(),
        'inference_time': t_infer,
        'annotated_image_name': a_name,
        'annotated_image_path': a_path,
        'predictions': predictions
        }
    return response

def warmup_image(batch_size, warmup_dim):
    for _ in range(batch_size):
        yield np.zeros([warmup_dim, warmup_dim, 3], dtype=np.uint8)

