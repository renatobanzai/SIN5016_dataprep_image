from datetime import datetime
import json
from os.path import isfile, join
from os import listdir
import numpy as np


start_time = datetime.now()
work_dir = "./datasets/hog/"

files = [f for f in listdir(work_dir) if isfile(join(work_dir, f))]

with open(work_dir + files[0]) as f:
    teste = json.load(f)
    lista = np.ndarray((len(files), len(teste)))

for file in files:
    with open(work_dir+file) as f:
        teste = json.load(f)

end_time = datetime.now()
diff = end_time - start_time

print("Rodou {} imagens em {} tempo".format(len(files), str(diff)))
