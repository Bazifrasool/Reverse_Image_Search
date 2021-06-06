#ensure db has only pictures
#ebsure set REVERSE_IMAGE_SEARCH_PATH

from pathlib import Path
import keras
import os
import h5py
import numpy
from tqdm import tqdm
import cv2
from scipy.spatial import distance
REVERSE_IMAGE_SEARCH_PATH = "NONE"




number_of_results =5
batch_size = 5
threads=batch_size
list_of_files = 0
n_files = -1
feature_emb = 0
file = 0
Path_of_database ="none"
batch_matr = []
model=0

def pre_proc(img):
    return cv2.resize(img,(224,224))
batch_matr = [x for x in range(0,n_files)]


def batch_preprocess(i):
    print(i , Path_of_database,str(Path_of_database / list_of_files[i]))
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
        return np.zeros(shape=(batch_size, 224, 224, 3),)
    batch_value=parallel_read(beg,end)
    return batch_value

def store_filename(l_o_f):
    with open(str(Path(REVERSE_IMAGE_SEARCH_PATH)/"model_files"/"dataset_names.csv"),"w+") as f:
        for i,each in enumerate(l_o_f):
            f.write(str(each)+","+str(i)+"\n")




class ReverseImageSearch:
    def __init__(self):
        self.model=keras.models.load_model(str(Path(REVERSE_IMAGE_SEARCH_PATH)/"model_files"/ "resnet50.h5"))
        self.model.Trainable = False
        print("Done init")

    def generate_feature_embeddings(self,db_path,batch_sz):
        global list_of_files,n_files,batch_matr,Path_of_database,batch_size
        batch_size=batch_sz
        Path_of_database = Path(db_path)
        list_of_files = os.listdir(Path_of_database)
        n_files=len(list_of_files)
        print( " Ensure Database has only readable image formats ")
        print(list_of_files)
        global file,feature_emb
        file = h5py.File(str(Path(REVERSE_IMAGE_SEARCH_PATH)/"model_files"/"feature_embeddings.h5"), "w")    
        feature_emb = file.create_dataset("features", (len(list_of_files)+batch_size,2048), h5py.h5t.IEEE_F32LE,compression="lzf")
        batch_matr = [x for x in range(0,n_files)]
        print(batch_matr)
        print("*****************-- INGESTING DATA--******************************")
        for index_i in tqdm(range(0,(n_files+batch_size//batch_size),batch_size)):
            batch=read_data(index_i,index_i+batch_size-1)
            print(index_i,batch,batch.shape)
            res=self.model.predict(batch)
            for i,each in enumerate(res):
                feature_emb[index_i+i,...]=each
        #print(each.shape,index_i+i,batch.shape)
     
        file.close()
        store_filename(list_of_files)
    
    def query_pic_class(self,path):
        img = cv2.imread(str(Path(path)))
        img = pre_proc(img)
        return (self.model.predict(img[None,...]))

    def calculate_distances_class(self,n,qpath):
        Path_of_query = qpath
        vector=self.query_pic_class(Path_of_query)
        ld = h5py.File(str(Path(REVERSE_IMAGE_SEARCH_PATH)/"model_files"/"feature_embeddings.h5"),"r")
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

    def query(self,n_results,query_path,db_path):
        Path_of_database = Path(db_path)
        number_of_results = n_results

        import pandas as pd
        dataset_filenames = pd.read_csv(str(Path(REVERSE_IMAGE_SEARCH_PATH)/"model_files"/"dataset_names.csv"),header=None)
        
        results=self.calculate_distances_class(number_of_results,query_path)
        #print(results)
        for i in results:
                print(str(Path_of_database / str(dataset_filenames.loc[i[0]][0])))
                #return str(Path_of_database / dataset_filenames.loc[results[i][0]][0])

import sys
if __name__ == "__main__":
    #Set folder path first
    #argv1 is path of database independet of folder location
    REVERSE_IMAGE_SEARCH_PATH = os.getcwd()
    #print(sys.argv)
    rv=ReverseImageSearch()
    rv.generate_feature_embeddings(sys.argv[1],4)
    rv.query(3,sys.argv[2],sys.argv[1])

