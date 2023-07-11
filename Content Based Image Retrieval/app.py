from flask import Flask
from flask import render_template
from flask import request
from werkzeug.utils import secure_filename
import os

from image_retrieval import operation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'E:\academic lab\CV\package\upload_folder'


allowed_extension = set(['png'])

def allowed_image_format( fileName ):
    return "." in fileName and fileName.split(".",1)[1] in allowed_extension 


@app.route("/" , methods = ['GET' , 'POST'])
def uploadPage():
    
    try:
        if request.method == 'POST' : 

            query_image = request.files['image']
            
            if query_image and allowed_image_format(query_image.filename):
                secured_filename = secure_filename("query_image.png") 
                query_image.save( os.path.join(app.config['UPLOAD_FOLDER'] , secured_filename ) )
                
                result  = operation()

                if not result :
                    return render_template( "uploadImage.html")

                else:
                    return render_template( "uploadImage.html" , image_urls = result)

        
        return render_template("uploadImage.html")
    
    except Exception as e :
        return render_template("exceptionPage.html" , name = e)


if __name__ == "__main__":
    app.run()