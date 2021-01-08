#fix batch in h5
#d_model.save("./resnet50.h5")
from tqdm import tqdm
from pathlib import Path

Path_of_database = Path("./data_base")
Path_of_query = str(Path("./") / "1.jpg")
features_of_database = "feature_embeddings.h5"
filenames_of_database = "data_set_names.csv"
number_of_results =5
batch_size = 1
threads=batch_size

import keras
model=keras.models.load_model("./resnet50.h5")
model.Trainable=False

import os
list_of_files = os.listdir(Path_of_database)
list_of_files = [ i for i in list_of_files if ".jpg" in i]
n_files=len(list_of_files)
#print(list_of_files)

import h5py
file = h5py.File(features_of_database, "w")    
feature_emb = file.create_dataset("features", (len(list_of_files),2048), h5py.h5t.IEEE_F32LE,compression="lzf")

import cv2
def pre_proc(img):
    return cv2.resize(img,(256,256))
batch_matr = [x for x in range(0,n_files)]

def batch_preprocess(i):
    matrix = cv2.imread(str(Path_of_database / list_of_files[i]))
    matrix = pre_proc(matrix)
    #print("that",matrix)
    global batch_matr
    batch_matr[i]=matrix

from concurrent.futures import ThreadPoolExecutor
import numpy as np
def parallel_read(begin,end):
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        for i in range(begin,end+1):
            executor.submit(batch_preprocess,(i))
    return np.array(batch_matr[begin:end+1],dtype='f')

def read_data(beg,end):
    if end>n_files:
        end=n_files-1
    batch_value=parallel_read(beg,end)
    return batch_value

for index_i in tqdm(range(0,n_files//batch_size,batch_size)):
    batch=read_data(index_i,index_i+batch_size-1)
    res=model.predict(batch)
    for i,each in enumerate(res):
        feature_emb[index_i+i,...]=each
file.close()

def store_filename(l_o_f):
    with open(filenames_of_database,"w+") as f:
        for i,each in enumerate(l_o_f):
            f.write(str(each)+","+str(i)+"\n")

store_filename(list_of_files)
import pandas as pd
dataset_filenames = pd.read_csv(filenames_of_database,header=None)

def query_pic(path):
    img = cv2.imread(path)
    img = pre_proc(img)
    return (model.predict(img[None,...]))
    
from scipy.spatial import distance
def calculate_distances(n):
    vector=query_pic(Path_of_query)
    ld = h5py.File(features_of_database,"r")
    values = []
    for each in ld["features"]:
        values.append((1-distance.cosine(each,vector)))
    from operator import itemgetter
    from heapq import nlargest
    return nlargest(n, enumerate(values), itemgetter(1))
results=calculate_distances(number_of_results)
import matplotlib.pyplot as plt

plt.figure()
fig=plt.figure(figsize=(8, 8))
columns = 1
rows = 5

for i in range(1, columns*rows +1):
    img=plt.imread(str(Path_of_database / dataset_filenames.loc[results[i-1][0]][0]))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()