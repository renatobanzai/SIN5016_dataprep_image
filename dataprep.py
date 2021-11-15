#!/usr/bin/env python3

import cv2
from skimage.feature import hog, local_binary_pattern
import json
from os.path import isfile, join
from os import listdir
from datetime import datetime
import numpy as np
from multiprocessing import Pool
import h5py
import pandas as pd



def get_face(image):
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
    faces = face_cascade.detectMultiScale(image)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = cv2.resize(image[y:y + h, x:x + w], (128,128))
        return face

def get_hog(image):
    feature_hog, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    # hog_image_rescaled = (exposure.rescale_intensity(hog_image, in_range=(0, 10)) *255).astype(np.uint8)
    return feature_hog

def get_lbp(image):
    radius = 3
    n_points = 8 * radius
    METHOD = "uniform"

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_lbp = local_binary_pattern(image, n_points, radius, METHOD)
    feature_lbp = cv2.calcHist(image_lbp.astype(np.uint8), [0], None, [256], (0, 256))
    return feature_lbp

def save(dest_dir, dest_file, value, dic_labels):
    with open(dest_dir+dest_file+".json", "w") as text_file:
        lista = np.ndarray.tolist(value)
        lista.insert(0, dic_labels[dest_file])
        json.dump(lista, text_file)

def dataprep_image(params):
    img_work_dir, file_name, dic_labels = params
    image = cv2.imread(img_work_dir+file_name)
    hog_dir = "./datasets/hog/"
    lbp_dir = "./datasets/lbp/"

    face = get_face(image)
    if not face is None:
        # feature_hog = get_hog(face)
        # save(hog_dir, file_name, feature_hog, dic_labels)
        feature_lbp = get_lbp(face)
        save(lbp_dir, file_name, feature_lbp, dic_labels)
    else:
        print("O arquivo {} nao processou vj.".format(file_name))

work_dir = "./images/celebA_30_percent/"
hog_dir = "./datasets/hog/"
lbp_dir = "./datasets/lbp/"

df_labels = pd.read_csv("./identity_CelebA.txt", sep=" ")
dic_labels = dict(df_labels.values)

imagens = [f for f in listdir(work_dir) if isfile(join(work_dir, f))]
max_iter = 100

start_time = datetime.now()

# Rodou 100 imagens em 0:00:07.546461 tempo
# for i in range(max_iter):
#     dataprep_image((work_dir, imagens[i]))

lista_params = []
for i in range(max_iter):
    lista_params.append((work_dir, imagens[i], dic_labels))

# Rodou 100 imagens em 0:00:02.271882 tempo
with Pool() as p:
    p.map(dataprep_image, lista_params)

end_time = datetime.now()
diff = end_time - start_time

print("Rodou {} imagens em {} tempo".format(max_iter, str(diff)))





