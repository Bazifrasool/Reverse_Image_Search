from pathlib import Path

Path_of_database = Path("./database")
Path_of_query = str(Path("./") / "query.jpg")
features_of_database = "./model_files/feature_embeddings.h5"
filenames_of_database = "./model_files/dataset_names.csv"
number_of_results =5
batch_size = 5
threads=batch_size
list_of_files = 0
n_files = -1
feature_emb = 0
file = 0
import keras
import os
import h5py
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import distance
batch_matr = []
model=keras.models.load_model("./model_files/resnet50.h5")
model.Trainable=False


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
    if end>=n_files:
        end=n_files-1
    if beg>=n_files:
        return np.zeros(shape=(batch_size, 256, 256, 3),)
    batch_value=parallel_read(beg,end)
    return batch_value

def store_filename(l_o_f):
    with open(filenames_of_database,"w+") as f:
        for i,each in enumerate(l_o_f):
            f.write(str(each)+","+str(i)+"\n")



def feature_embeddings_generate(path):
     global list_of_files,n_files,batch_matr,Path_of_database
     Path_of_database=Path(path)
     list_of_files = os.listdir(Path_of_database)
     list_of_files = [ i for i in list_of_files if ".jpg" in i]
     n_files=len(list_of_files)
     
     global file,feature_emb
     file = h5py.File(features_of_database, "w")    
     feature_emb = file.create_dataset("features", (len(list_of_files)+batch_size,2048), h5py.h5t.IEEE_F32LE,compression="lzf")
     batch_matr = [x for x in range(0,n_files)]
     
     for index_i in tqdm(range(0,(n_files+batch_size//batch_size),batch_size)):
         batch=read_data(index_i,index_i+batch_size-1)
         res=model.predict(batch)
         for i,each in enumerate(res):
             feature_emb[index_i+i,...]=each
        #print(each.shape,index_i+i,batch.shape)
     
     file.close()
     store_filename(list_of_files)

def query_pic(path):
    img = cv2.imread(path)
    img = pre_proc(img)
    return (model.predict(img[None,...]))

def calculate_distances(n,qpath):
    Path_of_query = qpath
    vector=query_pic(Path_of_query)
    ld = h5py.File(features_of_database,"r")
    values = []
    for each in ld["features"]:
        if(np.sum(each))<0.5:
            values.append(-999)
            break
        values.append((1-distance.cosine(each,vector)))
    #print(len(values))
    from operator import itemgetter
    from heapq import nlargest
    return nlargest(n, enumerate(values), itemgetter(1))

def query(n_results,query_path,gui=False):
    
    number_of_results = n_results
    query_path
    
    import pandas as pd
    dataset_filenames = pd.read_csv(filenames_of_database,header=None)
    
    results=calculate_distances(number_of_results,query_path)
    if(gui==False):
        for i in range(0,len(results)):
            print(str(Path_of_database / dataset_filenames.loc[results[i][0]][0]))
    else:
        plt.figure()
        fig=plt.figure(figsize=(8, 8))
        columns = 1
        rows = number_of_results

        for i in range(1, columns*rows +1):
            img=plt.imread(str(Path_of_database / dataset_filenames.loc[results[i-1][0]][0]))
            temp=fig.add_subplot(rows, columns, i)
            temp.title.set_text(str(Path_of_database / dataset_filenames.loc[results[i-1][0]][0]))
            plt.imshow(img)
        plt.tight_layout()
        plt.show()

     
     
     
     
     
     
     
if __name__ == "__main__":
    #feature_embeddings_generate("./database")
    print("done")
    query(2,"./q2.jpg",True)
   
        