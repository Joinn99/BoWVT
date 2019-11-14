import os
import numpy as np
import cv2

SIFT = cv2.xfeatures2d.SIFT_create()


def calculate_memory(file_dir, file_list, args):
    feature = int(0)
    for _, file in enumerate(file_list):
        descs, _ = sift(file_dir + os.sep + file, args.scale)
        feature += descs.shape[0]
        print('All feature numbers: {:8d} Memory: {:8d} KB'.format(
            feature, int(feature / 2)), flush=True, end='\r')
    return feature


def abs_norm(vec, norm):
    return np.linalg.norm(np.abs(vec), ord=norm, axis=1, keepdims=True)


def ransac(kpt_1, kpt_2, match):
    data_1 = np.array([kp.pt for kp in kpt_1])[match[0], :]
    data_2 = np.array([kp.pt for kp in kpt_2])[match[1], :]
    _, matches = cv2.findHomography(data_1, data_2, cv2.RANSAC, 20.0)
    return np.sum(matches)


def sift(img_path, scale):
    img = cv2.imread(img_path)
    img = cv2.resize(
        img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    kpt, descs = SIFT.detectAndCompute(img, None)
    return descs, kpt

def random_image(dirs):
    file_list = sorted([img for img in os.listdir(
        dirs) if img.endswith('.jpg') or img.endswith('.png')])
    image_id = np.random.randint(0, len(file_list))
    return str(dirs + os.sep + file_list[image_id]).replace('static/', '')
