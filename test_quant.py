import torch.nn as nn
import torch
import numpy as np
import onnx
import onnxruntime
from lib.backbone import StackedHGNetV1, efficientformerv2_s0
from conf import *

# -----------------------------------#
#   导出ONNX模型函数
# -----------------------------------#
def model_convert_onnx(model, input_shape, output_path):
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1])
    input_names = ["input1"]        # 导出的ONNX模型输入节点名称
    output_names = ["output1"]      # 导出的ONNX模型输出节点名称

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


if __name__ == '__main__':
    edge_info =  (
                (True, (0, 1, 2, 3, 4)),  # RightEyebrow
                (True, (5, 6, 7, 8, 9)),  # LeftEyebrow
                (False, (10, 11, 12, 13)),  # NoseLine
                (False, (14, 15, 16, 17, 18)),  # Nose
                (True, (19, 20, 21, 22, 23, 24)),  # RightEye
                (True, (25, 26, 27, 28, 29, 30)),  # LeftEye
                (True, (31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42)),  # OuterLip
                (True, (43, 44, 45, 46, 47, 48, 49, 50)),  # InnerLip
            )
    model = efficientformerv2_s0(pretrained=True, edge_info=edge_info)
    pretrained_weight = '/disk2/icml/STAR/ivslab/efficientformerv2_s0_0.0439/model/best_model.pkl'
    checkpoint = torch.load(pretrained_weight)
    model.load_state_dict(checkpoint["net_ema"], strict=False)
    # print(model)
    # 建议将模型转成 eval 模式
    model.eval()
    # 网络模型的输入尺寸
    input_shape = (256, 256)      
    # ONNX模型输出路径
    output_path = './MyNet.onnx'

    # 导出为ONNX模型
    model_convert_onnx(model, input_shape, output_path)
    print("model convert onnx finsh.")

    # -----------------------------------#
    #   复杂模型可以使用下面的方法进行简化   
    # -----------------------------------#
    # import onnxsim
    # MyNet_sim = onnxsim.simplify(onnx.load(output_path))
    # onnx.save(MyNet_sim[0], "MyNet_sim.onnx")

    # -----------------------------------------------------------------------#
    #   第一轮ONNX模型有效性验证，用来检查模型是否满足 ONNX 标准   
    #   这一步是必要的，因为无论模型是否满足标准，ONNX 都允许使用 onnx.save 存储模型，
    #   我们都不会希望生成一个不满足标准的模型~
    # -----------------------------------------------------------------------#
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("onnx model check_1 finsh.")

    # ----------------------------------------------------------------#
    #   第二轮ONNX模型有效性验证，用来验证ONNX模型与Pytorch模型的推理一致性   
    # ----------------------------------------------------------------#
    # 随机初始化一个模型输入，注意输入分辨率
    x = torch.randn(size=(1, 3, input_shape[0], input_shape[1]))
    # torch模型推理
    with torch.no_grad():
        torch_out = model(x)
    print(torch_out[2])            # tensor([[-0.5728,  0.1695, ..., -0.3256,  1.1357, -0.4081]])
    # print(type(torch_out))      # <class 'torch.Tensor'>

    # 初始化ONNX模型
    ort_session = onnxruntime.InferenceSession(output_path)
    # ONNX模型输入初始化
    ort_inputs = {ort_session.get_inputs()[0].name: x.numpy()}
    # ONNX模型推理
    ort_outs = ort_session.run(None, ort_inputs)
    # print(ort_outs)             # [array([[-0.5727689 ,  0.16947027,  ..., -0.32555276,  1.13574252, -0.40812433]], dtype=float32)]
    # print(type(ort_outs))       # <class 'list'>，里面是个numpy矩阵
    # print(type(ort_outs[0]))    # <class 'numpy.ndarray'>
    ort_outs = ort_outs[0]        # 把内部numpy矩阵取出来，这一步很有必要

    # print(torch_out.numpy().shape)      # (1, 10)
    # print(ort_outs.shape)               # (1, 10)

    # ----------------------------------------------------------------#
    # 比较实际值与期望值的差异，通过继续往下执行，不通过引发AssertionError
    # 需要两个numpy输入
    # ----------------------------------------------------------------#
    np.testing.assert_allclose(torch_out[2].numpy(), ort_outs, rtol=1e-03, atol=1e-05)
    print("onnx model check_2 finsh.")
    
import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch


TRT_LOGGER = trt.Logger() 
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem 
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream
        
def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="",\
               fp16_mode=False, int8_mode=False, save_engine=False,
              ):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network() as network,\
                trt.OnnxParser(network, TRT_LOGGER) as parser:
            
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 20
            builder.max_batch_size = max_batch_size
            #pdb.set_trace()
            config.set_flag(trt.BuilderFlag.FP16)  # Default: False
            config.set_flag(trt.BuilderFlag.INT8)  # Default: False
            if int8_mode:
                # To be updated
                raise NotImplementedError
                
            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))
                
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            #pdb.set_trace()
            #network.mark_output(network.get_layer(network.num_layers-1).get_output(0)) # Riz   
            #network.mark_output(network.get_layer(network.num_layers-1).get_output(1)) # Riz
            
            engine = builder.build_cuda_engine(network)
            if engine is None:
                print("Failed to create the engine")
                return None
            print("Completed creating Engine")
            
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine
        
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)
    
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

import os
import time

onnx_model_path = './MyNet.onnx'
pytorch_model_path = './model.pth'

# These two modes are dependent on hardwares
fp16_mode = False
int8_mode = True
trt_engine_path = './model_fp16_{}_int8_{}.trt'.format(fp16_mode, int8_mode)


max_batch_size = 1 # The batch size of input mush be smaller the max_batch_size once the engine is built
x_input = np.random.rand(max_batch_size, 3, 256, 256).astype(dtype=np.float32) 

# Build an engine
engine = get_engine(max_batch_size, onnx_model_path, trt_engine_path, fp16_mode, int8_mode)
# Please check that engine is not None
# If it is None, that means it failed to create the engine. 
# Create the context for this engine
context = engine.create_execution_context() 
# Allocate buffers for input and output
inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings 


# Do inference
shape_of_output = (max_batch_size, 10)
# Load data to the buffer
inputs[0].host = x_input.reshape(-1)
# inputs[1].host = ... for multiple input
t1 = time.time()
trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data
t2 = time.time()
output_from_trt_engine = postprocess_the_outputs(trt_outputs[0], shape_of_output)

# Compare with the PyTorch
# pth_model = CNN(10)
# pth_model.load_state_dict(torch.load(pytorch_model_path))
# pth_model.cuda()

# x_input_pth = torch.from_numpy(x_input).cuda()
# pth_model.export_to_onnx_mode = False
# t3 = time.time()
# output_from_pytorch_model =  pth_model(x_input_pth)
# t4 = time.time()
# output_from_pytorch_model = output_from_pytorch_model.cpu().data.numpy()


#mse = np.mean((output_from_trt_engine - output_from_pytorch_model)**2)
print("Inference time with the TensorRT engine: {}".format(t2-t1))
#print("Inference time with the PyTorch model: {}".format(t4-t3))
#print('MSE Error = {}'.format(mse))