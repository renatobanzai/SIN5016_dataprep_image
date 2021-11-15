import cv2
import matplotlib.pyplot as plt
import sys
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from skimage.io import imsave, imread
from os.path import isfile, join
from os import listdir
from datetime import datetime
from threading import Thread
from queue import Queue
from multiprocessing import Pool
import numpy as np

# classe para tornar possível threading
# class data_prep_hog(Thread):
#     def __init__(self, queue):
#         Thread.__init__(self)
#         self.queue = queue
#
#     def run(self):
#         while True:
#             img_work_dir, img_dest_dir, img_file_name = self.queue.get()
#             try:
#                 image = imread(img_work_dir + img_file_name)
#                 fd, hog_image = hog(image, orientations=9,
#                                     pixels_per_cell=(8, 8),
#                                     cells_per_block=(2, 2),
#                                     visualize=True)
#                 hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#                 imsave(img_dest_dir + img_file_name, hog_image_rescaled)
#             finally:
#                 self.queue.task_done()

# para rodar com threading
# q = Queue()
# for x in range(8):
#     prep = data_prep_hog(q)
#     prep.daemon = True
#     prep.start()
# não performou bem, quase dobrou o tempo com diferentes quantidades de threads (2 a 20)
# for i in range(max_iter):
#     q.put((work_dir, dest_dir, imagens[i]))
#
# q.join()


def processar_hog(params):
    img_work_dir = params[0]
    img_dest_dir = params[1]
    img_file_name = params[2]
    image = imread(img_work_dir+img_file_name)
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    hog_image_rescaled = (exposure.rescale_intensity(hog_image, in_range=(0, 10)) *255).astype(np.uint8)
    imsave(img_dest_dir+img_file_name, hog_image_rescaled)
    # salvar a variável fd num formato que seja facilmente legível: json, hdf5


# Rodou 100 imagens em 0:00:08.207836 tempo
# for i in range(max_iter):
#     processar_hog(work_dir, dest_dir, imagens[i])

work_dir = "./images/celebA_30_percent/vj_processado/"
dest_dir = "./images/celebA_30_percent/hog_processado/"

imagens = [f for f in listdir(work_dir) if isfile(join(work_dir, f))]
lista_params = []

max_iter = 97
iter = 0
for i in range(max_iter):
    lista_params.append((work_dir, dest_dir, imagens[i]))

start_time = datetime.now()

# Rodou 100 imagens em 0:00:08.207836 tempo
for i in range(max_iter):
    processar_hog((work_dir, dest_dir, imagens[i]))

# with Pool() as p:
#     p.map(processar_hog, lista_params)

end_time = datetime.now()
diff = end_time - start_time

print("Rodou {} imagens em {} tempo".format(max_iter, str(diff)))