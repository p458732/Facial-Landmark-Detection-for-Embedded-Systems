import torch.nn as nn
import torch
import numpy as np
import onnx
import onnxruntime
from lib.backbone import StackedHGNetV1, efficientformerv2_s0, mobile_vit_v2
from conf import *

# -----------------------------------#
#   导出ONNX模型函数
# -----------------------------------#
def model_convert_onnx(model, input_shape, output_path):
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1])
    input_names = ["input1"]        # 导出的ONNX模型输入节点名称
    output_names = ["output0", "output1", "output2", "output3"]      # 导出的ONNX模型输出节点名称

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=False,          # 如果指定为True，在导出的ONNX中会有详细的导出过程信息description
        keep_initializers_as_inputs=False,  # 若为True，会出现需要warning消除的问题
        opset_version=12,       # 版本通常为10 or 11
        input_names=input_names,
        output_names=output_names,
    )

from run_model import Alignment
import copy
import cv2
import dlib
from PIL import Image
import argparse
from facenet_pytorch import MTCNN
mtcnn = MTCNN(image_size=256, thresholds=[0.5,0.5,0.5],select_largest=True, keep_all=True, margin=0)
predictor_path =  './preprocess/shape_predictor_68_face_landmarks.dat'
sp = dlib.shape_predictor(predictor_path)
def dataset_gen():
    img_paths = []
    image_list_path = './image_list.txt'
    # load image paths
    with open(image_list_path, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            img_paths.append(line)

    # facial landmark detector
    args = argparse.Namespace()
    args.config_name = 'alignment'
    model_path = './checkpoint/best_model.pkl' 
    device_ids = [0] if torch.cuda.is_available() else [-1]
    alignment = Alignment(args, model_path, dl_framework="tf_lite", device_ids=device_ids)
    for face_file_path in img_paths:
        input_image = cv2.imread(face_file_path)
        image_draw = copy.deepcopy(input_image)
        detector = dlib.get_frontal_face_detector()
        dets = detector(input_image, 1)
        max_num_faces = 2
        dets = dets[:max_num_faces]
        
        # results = []
        imgg = Image.open(face_file_path)
        _, boxes = mtcnn(imgg)
        if boxes is None:
            print("Switch to dlib: ", face_file_path)
            for detection in dets:
                face = sp(input_image, detection)
                shape = []
                for i in range(68):
                    x = face.part(i).x
                    y = face.part(i).y
                    shape.append((x, y))
                shape = np.array(shape)
                x1, x2 = shape[:, 0].min(), shape[:, 0].max()
                y1, y2 = shape[:, 1].min(), shape[:, 1].max()
                scale = min(x2 - x1, y2 - y1) / 200 * 1.05
                center_w = (x2 + x1) / 2
                center_h = (y2 + y1) / 2

                scale, center_w, center_h = float(scale), float(center_w), float(center_h)
            yield [alignment.preprocess_tensorflow_lite(input_image, scale, center_w, center_h)[0]]
        else:
            boxes = boxes[:max_num_faces]
            if boxes.shape[0] > 1:
                n = boxes.shape[0]
                while n > 1:
                    n-=1
                    for i in range(n):        
                        if boxes[i][2] + boxes[i][0] < boxes[i + 1][2] + boxes[i + 1][0] :  
                            tmp = copy.copy(boxes[i])
                            boxes[i] = boxes[i + 1]
                            boxes[i + 1] = tmp
            for box in boxes:
                landmarks = np.array([[box[0], box[1]], [box[2], box[3]]])
                x1, x2 = landmarks[:, 0].min(), landmarks[:, 0].max()
                y1, y2 = landmarks[:, 1].min(), landmarks[:, 1].max()
                scale = min(x2 - x1, y2 - y1) / 200 * 1.05
                center_w = (x2 + x1) / 2
                center_h = (y2 + y1) / 2
                
                scale, center_w, center_h = float(scale), float(center_w), float(center_h)
                # landmarks_pv = alignment.analyze(input_image, scale, center_w, center_h)
                # results.append(landmarks_pv)
            yield [alignment.preprocess_tensorflow_lite(input_image, scale, center_w, center_h)[0]]




if __name__ == '__main__':
    # model = mobile_vit_v2()
    # pretrained_weight = '/disk2/icml/STAR/ivslab/mobile_vit_0.0496/model/best_model.pkl'
    # model.load_state_dict(torch.load(pretrained_weight)['net'], strict=False)
    # model.eval()
    # input_shape = (256,256)
    # output_path = './MyNet.onnx'
    # model_convert_onnx(model, input_shape, output_path)
    # print("model convert onnx finsh.")
    # onnx_model = onnx.load(output_path)
    # onnx.checker.check_model(onnx_model)
    # print("onnx model check_1 finsh.")
    # x = torch.randn(size=(1, 3, input_shape[0], input_shape[1]))
    # with torch.no_grad():
    #     torch_out = model(x)
    # ort_session = onnxruntime.InferenceSession(output_path)
    # ort_inputs = {ort_session.get_inputs()[0].name: x.numpy()}
    # ort_outs = ort_session.run(None, ort_inputs)
    # ort_outs = ort_outs[0]       
    # np.testing.assert_allclose(torch_out[2].numpy(), ort_outs, rtol=1e-03, atol=1e-05)
    # print("onnx model check_2 finsh.")
    
    
    #onnx.helper.printable_graph(onnx_model.graph)   
    # from onnx_tf.backend import prepare
    # output_path = './MyNet.onnx'
    # onnx_model = onnx.load(output_path)
    # tf_rep = prepare(onnx_model)
    # tf_rep.export_graph('./my')
    
    # import tensorflow as tf
    # tf.debugging.set_log_device_placement(True)
    # model = tf.saved_model.load('./my')
    # model.trainable = False

    # input_tensor = tf.random.uniform([1, 3, 256, 256])
    # out = model(**{'input1': input_tensor})
   #Print a Human readable representation of the graph

    #===============tf -> tflite================
    # import tensorflow as tf
    # converter = tf.lite.TFLiteConverter.from_saved_model('./my')
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TFLite ops
    #     tf.lite.OpsSet.SELECT_TF_OPS  # enable TF ops
    # ]
    # converter.allow_custom_ops = True
    # # enable float 16
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]
    # tflite_model = converter.convert()

    # # Save the model
    # with open('./my/converted_model.tflite', 'wb') as f:
    #     f.write(tflite_model)

    #===============tflite inference================
    # import numpy as np
    # import tensorflow as tf

    # # Load the TFLite model and allocate tensors
    # interpreter = tf.lite.Interpreter(model_path="./my/converted_model.tflite")
    # interpreter.allocate_tensors()

    # # Get input and output tensors
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    # # Test the model on random input data
    # input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    # interpreter.set_tensor(input_details[0]['index'], input_data)

    # interpreter.invoke()

    # # get_tensor() returns a copy of the tensor data
    # # use tensor() in order to get a pointer to the tensor
    # output_data = interpreter.get_tensor(output_details[1]['index'])
    # print(output_data)