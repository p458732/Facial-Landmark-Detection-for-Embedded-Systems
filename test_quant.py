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


if __name__ == '__main__':
    # model = mobile_vit_v2()
    # pretrained_weight = '/disk2/icml/STAR/ivslab/mobile_vit_0.0496/model/best_model.pkl'
    # model.load_state_dict(torch.load(pretrained_weight)['net'], strict=False)
    # model.eval()
    input_shape = (256,256)
    output_path = './MyNet.onnx'
    # model_convert_onnx(model, input_shape, output_path)
    # print("model convert onnx finsh.")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("onnx model check_1 finsh.")
    # x = torch.randn(size=(1, 3, input_shape[0], input_shape[1]))
    # with torch.no_grad():
    #     torch_out = model(x)
    # ort_session = onnxruntime.InferenceSession(output_path)
    # ort_inputs = {ort_session.get_inputs()[0].name: x.numpy()}
    # ort_outs = ort_session.run(None, ort_inputs)
    # ort_outs = ort_outs[0]       
    # np.testing.assert_allclose(torch_out[2].numpy(), ort_outs, rtol=1e-03, atol=1e-05)
    print("onnx model check_2 finsh.")
    onnx.helper.printable_graph(onnx_model.graph)   
    # from onnx_tf.backend import prepare

    # tf_rep = prepare(onnx_model)
    # tf_rep.export_graph('./my')
    
    import tensorflow as tf
    # tf.debugging.set_log_device_placement(True)
    # model = tf.saved_model.load('./my')
    # model.trainable = False

    # input_tensor = tf.random.uniform([1, 3, 256, 256])
    # out = model(**{'input1': input_tensor})
   # Print a Human readable representation of the graph

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

    

    # #Save the model
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