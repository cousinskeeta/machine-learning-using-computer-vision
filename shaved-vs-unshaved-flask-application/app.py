"""
Flask Documentation:     http://flask.pocoo.org/docs/
Jinja2 Documentation:    http://jinja.pocoo.org/2/documentation/
Werkzeug Documentation:  http://werkzeug.pocoo.org/documentation/

This file creates your application.
"""
from flask import Flask, flash, render_template, request, redirect, url_for
from flask import jsonify
import os
import base64
import numpy as np
import io
from PIL import Image
from keras.preprocessing.image import img_to_array
import pickle
## import libraries
# import keras  
# import tensorflow as tf
from keras import backend as K
from tensorflow import Graph, Session

app = Flask(__name__)

def join(source_dir, dest_file, read_size):
    # Create a new destination file
    output_file = open(dest_file, 'wb')
     
    # Get a list of the file parts
    parts = ['final_model1','final_model2','final_model3']
 
    # Go through each portion one by one
    for file in parts:
         
        # Assemble the full path to the file
        path = file
         
        # Open the part
        input_file = open(path, 'rb')
         
        while True:
            # Read all bytes of the part
            bytes = input_file.read(read_size)
             
            # Break out of loop if we are at end of file
            if not bytes:
                break
                 
            # Write the bytes to the output file
            output_file.write(bytes)
             
        # Close the input file
        input_file.close()
         
    # Close the output file
    output_file.close()
join(source_dir='', dest_file="Combined_Model.p", read_size = 50000000)
        
global model
graph1 = Graph()
with graph1.as_default():
	session1 = Session(graph=graph1)
	with session1.as_default():
		model = pickle.load(open('Combined_Model.p', 'rb'))


print(" * Loading application!")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image.reshape(1,300,300,3)
    print("Making prediction on image:::", type(image), image.shape)
    return image

@app.route("/", methods=["GET","POST"])
@app.route("/predictions", methods=["GET","POST"])
def predictions():
    return render_template("predict.html")

@app.route("/predict", methods=["GET","POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(300, 300))
    
    K.set_session(session1)
    with graph1.as_default():
        predictions = model.predict(processed_image)
        results = predictions[0][0]
        print(results)

    response = {
        'predictions': {
            'shaved': str(results * 100) + "%",
            'results': 'Shaved!' if results < 0.5 else 'Unshaved!'
        }
    }
    return jsonify(response)
if __name__ == '__main__':
    app.run(debug=True)