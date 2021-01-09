import api_class
import sys
if __name__ =="__main__":
    if len(sys.argv)==5 and sys.argv[1]=="query":
        qrie=api_class.ReverseImageSearch()
        print(qrie.query_class(int(sys.argv[2]),sys.argv[3],bool(sys.argv[4])))
    if len(sys.argv)==4 and sys.argv[1]=="gen":
        api_class.feature_embeddings_generate(sys.argv[2],int(sys.argv[3]))
    