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

work_dir = "./images/celebA_30_percent/"
dest_dir = work_dir + "vj_processado/"

def processar_viola_jones(params):
    img_work_dir = params[0]
    img_dest_dir = params[1]
    img_file_name = params[2]

    image = cv2.imread(img_work_dir+img_file_name)

    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # precisa ser escala de cinza?

    faces = face_cascade.detectMultiScale(image)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = cv2.resize(image[y:y + h, x:x + w], 128,128)
        cv2.imwrite(dest_dir + img_file_name, face)

# processar_viola_jones((work_dir, dest_dir, "000002.jpg"))


max_iter = 100


imagens = [f for f in listdir(work_dir) if isfile(join(work_dir, f))]

start_time = datetime.now()

# Rodou 100 imagens em 0:00:02.605249 tempo
# for i in range(max_iter):
#     processar_viola_jones((work_dir, dest_dir, imagens[i]))

lista_params = []
for i in range(max_iter):
    lista_params.append((work_dir, dest_dir, imagens[i]))


# Rodou 100 imagens em 0:00:01.207836 tempo
# with Pool() as p:
#     p.map(processar_viola_jones, lista_params)

end_time = datetime.now()
diff = end_time - start_time
print("Rodou {} imagens em {} tempo".format(max_iter, str(diff)))

# lista_params = []
#
# max_iter = 100
# iter = 0
# for i in range(max_iter):
#     lista_params.append((work_dir, dest_dir, imagens[i]))
#
# start_time = datetime.now()

# q = Queue()
# for x in range(8):
#     prep = data_prep_hog(q)
#     prep.daemon = True
#     prep.start()
#
# for i in range(max_iter):
#     q.put((work_dir, dest_dir, imagens[i]))
#
# q.join()



# with Pool() as p:
#     p.map(processar_hog, lista_params)
#
#
# end_time = datetime.now()
# diff = end_time - start_time
# print("Rodou {} imagens em {} tempo".format(max_iter, str(diff)))
