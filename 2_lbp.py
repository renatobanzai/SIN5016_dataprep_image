import cv2
from skimage.feature import local_binary_pattern
from skimage import data, exposure
from skimage.io import imsave, imread
from os.path import isfile, join
from os import listdir
from datetime import datetime

import numpy as np


def processar_lbp(params):
    img_work_dir = params[0]
    img_dest_dir = params[1]
    img_file_name = params[2]

    radius = 3
    n_points = 8 * radius
    METHOD = "uniform"

    image = imread(img_work_dir+img_file_name, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_lbp = local_binary_pattern(image, n_points, radius, METHOD)
    feature_lbp = cv2.calcHist(image_lbp.astype(np.uint8), [0], None, [256], (0, 256))
    feature_lbp.squeeze().astype(np.uint8)
    imsave(img_dest_dir+img_file_name, feature_lbp.squeeze().astype(np.uint8))
    # salvar a variável fd num formato que seja facilmente legível: json, hdf5


# Rodou 100 imagens em 0:00:08.207836 tempo
# for i in range(max_iter):
#     processar_hog(work_dir, dest_dir, imagens[i])

work_dir = "./images/celebA_30_percent/vj_processado/"
dest_dir = "./images/celebA_30_percent/lbp_processado/"

imagens = [f for f in listdir(work_dir) if isfile(join(work_dir, f))]
lista_params = []

max_iter = 97
iter = 0
for i in range(max_iter):
    lista_params.append((work_dir, dest_dir, imagens[i]))

start_time = datetime.now()

# Rodou 100 imagens em 0:00:08.207836 tempo
for i in range(max_iter):
    processar_lbp((work_dir, dest_dir, imagens[i]))

# with Pool() as p:
#     p.map(processar_hog, lista_params)

end_time = datetime.now()
diff = end_time - start_time

print("Rodou {} imagens em {} tempo".format(max_iter, str(diff)))