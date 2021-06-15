from flask import Flask,render_template,request,url_for,send_from_directory
from flask_bootstrap import Bootstrap
import os
from werkzeug.utils import secure_filename
from model_api_class import ReverseImageSearch
from pathlib import Path
import os
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
MEDIA_FOLDER ="./database"
app = Flask(__name__)
app.config["UPLOAD_FOLDER"]=UPLOAD_FOLDER
app.config["MEDIA"]= MEDIA_FOLDER

Bootstrap(app)
gbGenerateEmbeddings = False
#initialize model and generate

# create object
# generate features

ris = None


@app.route("/")
def index():
    global ris
    ris=ReverseImageSearch(os.getcwd())
    if(os.environ['GENEMBED']=="TRUE"):
        ris.generate_feature_embeddings("./database",1)
    return render_template("base.html")

@app.route("/<path:filename>")
def img_send(filename):
    return send_from_directory(MEDIA_FOLDER,filename)

#must return a list
def get_results(num,filename):
    print(num,filename)
    lst=ris.query(num,str(Path("./uploads")/filename),"/database")
    print(lst)
    # pass from ./uploads/filename and num into query function and return a list of fns from db
    return lst

@app.route("/processing",methods=["GET","POST"])
def processing():
    if request.method == 'POST':
        
        if 'img' not in request.files:
            # flash('No file part')
            return "Missing"
        file = request.files['img']
        
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            # flash('No selected file')
            return "Not Selected"
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    
        lst=get_results(int(request.form.get("num")),filename)

 
        
        return render_template("results.html",results=lst)

if __name__=="__main__":
     app.run(debug=True,port=7000,host="0.0.0.0")