#importing the necessary libraries
from flask import Flask , jsonify
from flask import render_template
from flask import request


#importing the find_similar_college function from the findSimilarCollege.py 
from findSimilarCollege import find_similar_college


#creating a instance of Flask object
app = Flask(__name__)


#initial page route which is being executed once app starts
@app.route("/" , methods = ['GET' , 'POST'])
def uploadPage():
    
    #executing a try block
    try:

        #if request method is post
        if request.method == 'POST' :
            
            #retreiving the query value from the form
            query   = request.form['query']

            #making a call to find_similar_college by passing the query as the parameter 
            #receving the result in the form of dataframe. 
            #hence converting it into dictionary using to_dict function from pandas. 
            #orient='records' s used to create a list from the dictionary
            result  = find_similar_college(query).to_dict(orient='records')

            #if the result is empty : []
            if not result :
                return render_template( "queryUploaded.html")
            
            #else passing the result value to data  to the userInterface.html
            else:
                return render_template( "userInferface.html" , data = result )

        #if request method is not post passing the result value as empty [] to data to the userInterface.html
        return render_template("userInferface.html" , data = [])
    
    #if exception occured it is shown in a separate page at exceptionPage.html
    except Exception as e :
        return render_template("exceptionPage.html" , name = e)


#main function
if __name__ == "__main__":
    
    #flask app is being executed
    app.run()