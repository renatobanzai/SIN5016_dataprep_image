from datetime import datetime
import json
from os.path import isfile, join
from os import listdir
import numpy as np
import h5py
import pandas as pd

start_time = datetime.now()
work_dir = "./datasets/hog/"
file_hd5 = "./datasets/hog_{}".format(datetime.now().strftime("%m_%d_%H_%M"))




files = [f for f in listdir(work_dir) if isfile(join(work_dir, f))]



with open(work_dir + files[0]) as f:
    teste = json.load(f)
    lista = np.ndarray((len(files), len(teste)))



i = 0
try:
    h5 = h5py.File(file_hd5, "w")
    dataset = h5.create_dataset("descriptor", lista.shape)
    # Rodou 193868 imagens em 0:10:19.006672 tempo
    for file in files:
            try:
                with open(work_dir+file) as f:
                    teste = json.load(f)
                    dataset[i] = teste
                    i += 1
            except:
                pass
except:
    pass
finally:
    h5.close()


curl -X POST -L \
    -H "Authorization: Bearer ya29.a0ARrdaM-BFmuhh07G7E55EjDFCSMf-vLqWByvhPi1RTyGPcdR7JnV5K8oemIy6pj6h8_WcHJTr4la6LmTUBAWLr2MFZ2F35K0ZBEun9D8ctxjL6RtQpoOfOQ5PjDnIjaWLI7e-HpGNAUIMNewhXeafWW87Kgl" \
    -F "metadata={name :'lbp_11_15_18_20'};type=application/json;charset=UTF-8" \
    -F "file=@lbp_11_15_18_20;type=application/zip" \
    "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"


end_time = datetime.now()
diff = end_time - start_time

print("Rodou {} imagens em {} tempo".format(len(files), str(diff)))
