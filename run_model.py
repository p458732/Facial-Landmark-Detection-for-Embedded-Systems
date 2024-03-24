import glob
import sys
import os
import cv2
import copy
import dlib
import math
from tensorflow import convert_to_tensor, float32, transpose, squeeze, constant
import tflite_runtime.interpreter as tflite
import argparse
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torch
from facenet_pytorch import MTCNN
# private package
from lib import utility


mtcnn = MTCNN(image_size=256, thresholds=[0.5,0.5,0.5],select_largest=True, keep_all=True, margin=0)

class GetCropMatrix():
    """
    from_shape -> transform_matrix
    """

    def __init__(self, image_size, target_face_scale, align_corners=False):
        self.image_size = image_size
        self.target_face_scale = target_face_scale
        self.align_corners = align_corners

    def _compose_rotate_and_scale(self, angle, scale, shift_xy, from_center, to_center):
        cosv = math.cos(angle)
        sinv = math.sin(angle)

        fx, fy = from_center
        tx, ty = to_center

        acos = scale * cosv
        asin = scale * sinv

        a0 = acos
        a1 = -asin
        a2 = tx - acos * fx + asin * fy + shift_xy[0]

        b0 = asin
        b1 = acos
        b2 = ty - asin * fx - acos * fy + shift_xy[1]

        rot_scale_m = np.array([
            [a0, a1, a2],
            [b0, b1, b2],
            [0.0, 0.0, 1.0]
        ], np.float32)
        return rot_scale_m

    def process(self, scale, center_w, center_h):
        if self.align_corners:
            to_w, to_h = self.image_size - 1, self.image_size - 1
        else:
            to_w, to_h = self.image_size, self.image_size

        rot_mu = 0
        scale_mu = self.image_size / (scale * self.target_face_scale * 200.0)
        shift_xy_mu = (0, 0)
        matrix = self._compose_rotate_and_scale(
            rot_mu, scale_mu, shift_xy_mu,
            from_center=[center_w, center_h],
            to_center=[to_w / 2.0, to_h / 2.0])
        return matrix


class TransformPerspective():
    """
    image, matrix3x3 -> transformed_image
    """

    def __init__(self, image_size):
        self.image_size = image_size

    def process(self, image, matrix):
        return cv2.warpPerspective(
            image, matrix, dsize=(self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR, borderValue=0)


class TransformPoints2D():
    """
    points (nx2), matrix (3x3) -> points (nx2)
    """

    def process(self, srcPoints, matrix):
        # nx3
        desPoints = np.concatenate([srcPoints, np.ones_like(srcPoints[:, [0]])], axis=1)
        desPoints = desPoints @ np.transpose(matrix)  # nx3
        desPoints = desPoints[:, :2] / desPoints[:, [2, 2]]
        return desPoints.astype(srcPoints.dtype)


class Alignment:
    def __init__(self, args, model_path, dl_framework, device_ids):
        self.input_size = 256
        self.target_face_scale = 1.0
        self.dl_framework = dl_framework
        self.time = 0

        # model
        if self.dl_framework == "pytorch":
            # conf
            self.config = utility.get_config(args)
            self.config.device_id = device_ids[0]
            # set environment
            utility.set_environment(self.config)
            self.config.init_instance()
            if self.config.logger is not None:
                
                self.config.logger.info("Loaded configure file %s: %s" % (args.config_name, self.config.id))
                self.config.logger.info("\n" + "\n".join(["%s: %s" % item for item in self.config.__dict__.items()]))

            net = utility.get_net(self.config)
            if device_ids == [-1]:
                checkpoint = torch.load(model_path, map_location="cpu")
            else:
                checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint["net"])
            net = net.to(self.config.device_id)
            net.eval()
            self.alignment = net
        elif self.dl_framework == "tf":
            import tensorflow as tf
            self.config = utility.get_config(args)
            self.config.device_id = device_ids[0]
            
            model = tf.saved_model.load('./my')
            model.trainable = False
            self.alignment = model
            
        elif self.dl_framework == "tf_lite":
            self.config = utility.get_config(args)
            self.config.device_id = device_ids[0]
            
            interpreter = tflite.Interpreter(model_path="./best.tflite")
            interpreter.allocate_tensors()
            # Get input and output tensors
            self.input_details = interpreter.get_input_details()
            self.output_details = interpreter.get_output_details()
            
            self.alignment = interpreter
            
            pass
        else:
            assert False

        self.getCropMatrix = GetCropMatrix(image_size=self.input_size, target_face_scale=self.target_face_scale,
                                           align_corners=True)
        self.transformPerspective = TransformPerspective(image_size=self.input_size)
        self.transformPoints2D = TransformPoints2D()

    def norm_points(self, points, align_corners=False):
        if align_corners:
            # [0, SIZE-1] -> [-1, +1]
            return points / torch.tensor([self.input_size - 1, self.input_size - 1]).to(points).view(1, 1, 2) * 2 - 1
        else:
            # [-0.5, SIZE-0.5] -> [-1, +1]
            return (points * 2 + 1) / torch.tensor([self.input_size, self.input_size]).to(points).view(1, 1, 2) - 1

    def denorm_points(self, points, align_corners=False):
        if self.dl_framework == "pytorch":
            if align_corners:
                # [-1, +1] -> [0, SIZE-1]
                landmarks = (points + 1) / 2 * torch.tensor([self.input_size - 1, self.input_size - 1]).to(points).view(1, 1, 2)
                landmarks = landmarks.data.cpu().numpy()[0]
                return landmarks
            else:
                # [-1, +1] -> [-0.5, SIZE-0.5]
                landmarks = ((points + 1) * torch.tensor([self.input_size, self.input_size]).to(points).view(1, 1, 2) - 1) / 2
                landmarks = landmarks.data.cpu().numpy()[0]
                return landmarks
            
        elif self.dl_framework == "tf" or self.dl_framework == "tf_lite":
            if align_corners:
                # [-1, +1] -> [0, SIZE-1]
                landmarks = (points + 1) / 2 * constant([self.input_size - 1, self.input_size - 1], dtype=float32)
                landmarks = squeeze(landmarks).numpy()
                return landmarks
            else:
                # [-1, +1] -> [-0.5, SIZE-0.5]
                landmarks = ((points + 1) * constant([self.input_size, self.input_size], dtype=float32) - 1) / 2
                landmarks = squeeze(landmarks).numpy()
                return landmarks


    def preprocess_pytorch(self, image, scale, center_w, center_h):
        matrix = self.getCropMatrix.process(scale, center_w, center_h)
        input_tensor = self.transformPerspective.process(image, matrix)
        input_tensor = input_tensor[np.newaxis, :]

        input_tensor = torch.from_numpy(input_tensor)
        input_tensor = input_tensor.float().permute(0, 3, 1, 2)
        input_tensor = input_tensor / 255.0 * 2.0 - 1.0
        input_tensor = input_tensor.to(self.config.device_id)
        return input_tensor, matrix
    
    def preprocess_tensorflow(self, image, scale, center_w, center_h):
        matrix = self.getCropMatrix.process(scale, center_w, center_h)
        input_tensor = self.transformPerspective.process(image, matrix)
        input_tensor = input_tensor[np.newaxis, :]

        input_tensor = convert_to_tensor(input_tensor, dtype=float32)
        input_tensor = transpose(input_tensor, perm=[0, 3, 1, 2])
        input_tensor = input_tensor / 255.0 * 2.0 - 1.0

        #input_tensor = input_tensor.to(self.config.device_id)
        return input_tensor, matrix
    
    def preprocess_tensorflow_lite(self, image, scale, center_w, center_h):
        matrix = self.getCropMatrix.process(scale, center_w, center_h)
        input_tensor = self.transformPerspective.process(image, matrix)
        input_tensor = input_tensor[np.newaxis, :]

        input_tensor = convert_to_tensor(input_tensor, dtype=float32)
        input_tensor = transpose(input_tensor, perm=[0, 3, 1, 2])
        input_tensor = input_tensor / 255.0 * 2.0 - 1.0
        
        #input_tensor = input_tensor.to(self.config.device_id)
        return input_tensor, matrix
    
    def postprocess(self, srcPoints, coeff):
        # dstPoints = self.transformPoints2D.process(srcPoints, coeff)
        # matrix^(-1) * src = dst
        # src = matrix * dst
        dstPoints = np.zeros(srcPoints.shape, dtype=np.float32)
        for i in range(srcPoints.shape[0]):
            dstPoints[i][0] = coeff[0][0] * srcPoints[i][0] + coeff[0][1] * srcPoints[i][1] + coeff[0][2]
            dstPoints[i][1] = coeff[1][0] * srcPoints[i][0] + coeff[1][1] * srcPoints[i][1] + coeff[1][2]
        return dstPoints

    def analyze(self, image, scale, center_w, center_h):
        if self.dl_framework == "pytorch":
            input_tensor, matrix = self.preprocess_pytorch(image, scale, center_w, center_h)
            with torch.no_grad():
                output = self.alignment(input_tensor)
            landmarks = output[2][0]
            
        elif self.dl_framework == "tf":
            input_tensor, matrix = self.preprocess_tensorflow(image, scale, center_w, center_h)

            out = self.alignment(**{'input1': input_tensor})
            landmarks = out['output0']
            pass
        elif self.dl_framework == "tf_lite":
            input_tensor, matrix = self.preprocess_tensorflow_lite(image, scale, center_w, center_h)
            self.alignment.set_tensor(self.input_details[0]['index'], input_tensor)
            self.alignment.invoke()
            landmarks = self.alignment.get_tensor(self.output_details[1]['index'])
        else:
            assert False

        landmarks = self.denorm_points(landmarks)
        
        landmarks = self.postprocess(landmarks, np.linalg.inv(matrix))

        return landmarks


def get_two_faces_list():
    l = []
    with open('./annotations/ivslab/test_q.txt', 'r') as f:
        l = f.readlines()
    return l
def process(input_image, path=None):
    
    image_draw = copy.deepcopy(input_image)
    dets = detector(input_image, 1)
    max_num_faces = 2
    dets = dets[:max_num_faces]
    
    results = []
    imgg = Image.open(path)
    _, boxes = mtcnn(imgg)
    if boxes is None:
        print("Switch to dlib: ", path)
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
            landmarks_pv = alignment.analyze(input_image, scale, center_w, center_h)
            results.append(landmarks_pv)
            
        return None, results
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
    for box  in boxes:
        landmarks = np.array([[box[0], box[1]], [box[2], box[3]]])
        x1, x2 = landmarks[:, 0].min(), landmarks[:, 0].max()
        y1, y2 = landmarks[:, 1].min(), landmarks[:, 1].max()
        scale = min(x2 - x1, y2 - y1) / 200 * 1.05
        center_w = (x2 + x1) / 2
        center_h = (y2 + y1) / 2
        
        scale, center_w, center_h = float(scale), float(center_w), float(center_h)
        landmarks_pv = alignment.analyze(input_image, scale, center_w, center_h)
        results.append(landmarks_pv)
    return None, results


if __name__ == '__main__':
    img_paths = []
    # sys.argv[1] = './image_list.txt'
    # sys.argv[2] = './test_out'
    image_list_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # load image paths
    with open(image_list_path, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            img_paths.append(line)
   
    # load face detector 
    predictor_path =  './preprocess/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    
    # facial landmark detector
    args = argparse.Namespace()
    args.config_name = 'alignment'
    model_path = './ivslab/mobile_vit_0.0496/model/best_model.pkl'
    device_ids = [0] if torch.cuda.is_available() else [-1]
    alignment = Alignment(args, model_path, dl_framework="tf_lite", device_ids=device_ids)
        
    #two_faces_list = get_two_faces_list()
    for face_file_path in img_paths:
        image = cv2.imread(face_file_path)
        image_draw, results = process(image, face_file_path)
        
        with open (f'{output_path}/{face_file_path.split("/")[-1].split(".")[0]}.txt','w') as f:
            for result in results:
                f.write('version: 1\n' + 'n_points: 51\n' + '{\n')
                for landmark in result:
                    f.write(f"{landmark[0]:.3f}" + ' ' + f"{landmark[1]:.3f}" + '\n')
                f.write('}\n')
       

    # demo
    # interface = gr.Interface(fn=process, inputs="image", outputs="image")
    # interface.launch(share=True)
