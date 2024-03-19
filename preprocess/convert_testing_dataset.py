import math
import cv2
import dlib
import numpy as np
from PIL import Image, ImageOps
import glob
from facenet_pytorch import MTCNN, InceptionResnetV1

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size=256, margin=0)


mode = ""
final_round = True
if final_round:
    mode=""
else:
    mode="ivslab_facial_test_private_qualification"
MODEL_PATH = "./preprocess/shape_predictor_68_face_landmarks.dat"
img_paths = sorted(glob.glob(f"./images/{mode}/*.png"))
detector = dlib.get_frontal_face_detector()


pts = []

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

def get_face_landmarks(image_path):
    # Load the image
    image = cv2.imread(image_path)
    try:
        image = ImageOps.exif_transpose(image)
    except:
        print("exif problem, not rotating")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize dlib's facial landmarks predictor
    predictor = dlib.shape_predictor("./preprocess/shape_predictor_68_face_landmarks.dat")  

    # Detect faces in the image
    faces = detector(gray)

    if len(faces) > 0:
        # Assume the first face is the target, you can modify this based on your requirements
        shape = predictor(gray, faces[0])
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        return landmarks
    else:
        return None

def calculate_roll_and_yaw(landmarks):
    # Calculate the roll angle using the angle between the eyes
    roll_angle = np.degrees(np.arctan2(landmarks[1, 1] - landmarks[0, 1], landmarks[1, 0] - landmarks[0, 0]))

    # Calculate the yaw angle using the angle between the eyes and the tip of the nose
    yaw_angle = np.degrees(np.arctan2(landmarks[1, 1] - landmarks[2, 1], landmarks[1, 0] - landmarks[2, 0]))

    return roll_angle, yaw_angle

def detect_and_crop_head(input_image, landmarks,factor=1.05):
    # Get facial landmarks
    
    if landmarks is not None:
        # Calculate the center of the face using the mean of the landmarks
        x1, x2 = landmarks[:, 0].min(), landmarks[:, 0].max()
        y1, y2 = landmarks[:, 1].min(), landmarks[:, 1].max()
        scale = min(x2 - x1, y2 - y1) / 200 * 1.05
        center_w = (x2 + x1) / 2
        center_h = (y2 + y1) / 2

        scale, center_w, center_h = float(scale), float(center_w), float(center_h)
        getCropMatrix = GetCropMatrix(image_size=256, target_face_scale=1.0,
                                           align_corners=True)
        cropping_matrix = getCropMatrix.process(scale, center_w, center_h)
        transformPerspective = TransformPerspective(image_size=256)
        input_tensor = transformPerspective.process(input_image, cropping_matrix)
        
        # Return the cropped head image
        return input_tensor
    else:
        return None

def get_landmarks(landmarks_path):
    pts = []
    with open(landmarks_path, 'r') as f:
        for line in f.readlines():
            if line[0] >= '0' and line[0] <= '9':
                x, y = float(line.split(' ')[0]), float(line.split(' ')[1])
                pts.append([x,y])
    return pts

def generate_metadata():
    not_detected_faces = 0
    prev_landmarks_68_str = ""
    with open('./annotations/ivslab/islab_test_q.tsv', 'w') as f2:
            for (idx, img_path) in enumerate(img_paths):
                file_name = img_path
                # handle pre_landmarks
                print("cur img: ", img_path, " not_detected_face_count: ", not_detected_faces)
                landmarks = get_face_landmarks(img_path)
                if landmarks is None:
                    # switch to mtcnn
                    not_detected_faces += 1
                    img = Image.open(img_path)
                    _, boxes = mtcnn(img)
                    if boxes is None:
                        continue
                    boxes = boxes[0]
                    landmarks = np.array([[boxes[0], boxes[1]], [boxes[2], boxes[3]]])
                    x1, x2 = landmarks[:, 0].min(), landmarks[:, 0].max()
                    y1, y2 = landmarks[:, 1].min(), landmarks[:, 1].max()
                    scale = min(x2 - x1, y2 - y1) / 200 * 1.05
                    center_w = (x2 + x1) / 2
                    center_h = (y2 + y1) / 2

                    scale, center_w, center_h = str(float(scale)), str(float(center_w)), str(float(center_h))
                    f2.write(file_name + "\t" + prev_landmarks_68_str + "\t" + prev_landmarks_68_str + "\t" + scale + "\t" + center_w + "\t" + center_h + "\n")
                    continue
                landmarks_str = []
                landmarks_68_str = ''
                for landmark in landmarks:
                    landmarks_str.append(','.join([str(x) for x in landmark]))
                landmarks_68_str = ','.join(landmarks_str)
                prev_landmarks_68_str = landmarks_68_str
                
                # handle gt_landmarks
                
                # handle scale, center
                landmarks = np.array(landmarks)
    
                if landmarks is not None:
                    # Calculate the center of the face using the mean of the landmarks
                    x1, x2 = landmarks[:, 0].min(), landmarks[:, 0].max()
                    y1, y2 = landmarks[:, 1].min(), landmarks[:, 1].max()
                    scale = min(x2 - x1, y2 - y1) / 200 * 1.05
                    center_w = (x2 + x1) / 2
                    center_h = (y2 + y1) / 2

                    scale, center_w, center_h = str(float(scale)), str(float(center_w)), str(float(center_h))
                f2.write(file_name + "\t" + landmarks_68_str + "\t" + landmarks_68_str + "\t" + scale + "\t" + center_w + "\t" + center_h + "\n")

if __name__ == '__main__':
    generate_metadata()