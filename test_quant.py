from run_model import *
from lib import utility
import argparse
from PIL import Image
import dlib
import cv2
import copy
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
    output_names = ["output0", "output1",
                    "output2", "output3"]      # 导出的ONNX模型输出节点名称

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=False,          # 如果指定为True，在导出的ONNX中会有详细的导出过程信息description
        keep_initializers_as_inputs=False,  # 若为True，会出现需要warning消除的问题
        opset_version=11,       # 版本通常为10 or 11
        input_names=input_names,
        output_names=output_names,
    )




def dataset_gen():
    # facial landmark detector
    args = argparse.Namespace()
    args.config_name = 'alignment'
    model_path = './ivslab/stackedHGnet_v1_0.0378/model/best_model.pkl' 
    device_ids = [0] if torch.cuda.is_available() else [-1]
    alignment = Alignments(args, model_path, dl_framework="pytorch", device_ids=device_ids)
    config = alignment.config

    metadata_path = "./annotations/ivslab/train.tsv"
    with open(metadata_path, 'r') as f:
        lines = f.readlines()
    
    cnt = 0
    for k, line in enumerate(lines):
        cnt += 1
        item = line.strip().split("\t")
        image_name, landmarks_5pts, landmarks_gt, scale, center_w, center_h = item[:6]
        # image & keypoints alignment
        image_name = image_name.replace('\\', '/')
        image_name = image_name.replace('//msr-facestore/Workspace/MSRA_EP_Allergan/users/yanghuan/training_data/wflw/rawImages/', '')
        image_name = image_name.replace('./rawImages/', '')
        # image_path = os.path.join(config.image_dir, image_name)
        landmarks_gt = np.array(list(map(float, landmarks_gt.split(","))), dtype=np.float32).reshape(-1, 2)
        scale, center_w, center_h = float(scale), float(center_w), float(center_h)
        input_image = cv2.imread(image_name)

        yield [alignment.preprocess_tensorflow_lite(input_image, scale, center_w, center_h)[0]]

edge_info = (
                (True, (0, 1, 2, 3, 4)),  # RightEyebrow
                (True, (5, 6, 7, 8, 9)),  # LeftEyebrow
                (False, (10, 11, 12, 13)),  # NoseLine
                (False, (14, 15, 16, 17, 18)),  # Nose
                (True, (19, 20, 21, 22, 23, 24)),  # RightEye
                (True, (25, 26, 27, 28, 29, 30)),  # LeftEye
                (True, (31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42)),  # OuterLip
                (True, (43, 44, 45, 46, 47, 48, 49, 50)),  # InnerLip
            )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Entry Function")
    group = parser.add_argument_group('train')
    group.add_argument("--train_num_workers", type=int, default=None, help="the num of workers in train process")
    #args = parser.parse_args()
    a = "s"
    config = utility.get_config(a)
    model = mobile_vit_v2()
    pretrained_weight = './checkpoint/best_model.pkl'
    from onnxsim import simplify
    model.load_state_dict(torch.load(pretrained_weight)['net'], strict=False)
    model.eval()
    input_shape = (256, 256)
    output_path = './MyNet.onnx'
    model_convert_onnx(model, input_shape, output_path)    
    print("model convert onnx finsh.")
    onnx_model = onnx.load(output_path)
    model_simp, check = simplify(onnx_model)
    onnx.save(model_simp, output_path)
    onnx_model = onnx.load(output_path)
    print('finished exporting onnx')
    print("onnx model check_1 finsh.")
    x = torch.randn(size=(1, 3, input_shape[0], input_shape[1]))
    with torch.no_grad():
        torch_out = model(x)
    ort_session = onnxruntime.InferenceSession(output_path)
    ort_inputs = {ort_session.get_inputs()[0].name: x.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_outs = ort_outs[0]
    np.testing.assert_allclose(
        torch_out[2].numpy(), ort_outs, rtol=1e-03, atol=1e-05)
    print("onnx model check_2 finsh.")

    # ===============onnx -> tf================
    from onnx_tf.backend import prepare
    output_path = './MyNet.onnx'
    onnx_model = onnx.load(output_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph('./my')

    import tensorflow as tf
    tf.debugging.set_log_device_placement(True)
    model = tf.saved_model.load('./my')
    model.trainable = False

    input_tensor = tf.random.uniform([1, 3, 256, 256])
    out = model(**{'input1': input_tensor})
 

    # ===============tf -> tflite================
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model('./my')
    tflite_model = converter.convert()

    # enable dynamic range
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float16]
    # converter.experimental_new_converter = True
    # converter.experimental_new_quantizer = True
    # converter.representative_dataset = dataset_gen
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_model = converter.convert()

    # Save the model
    with open('./best.tflite', 'wb') as f:
        f.write(tflite_model)

    # ===============tflite inference================
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
