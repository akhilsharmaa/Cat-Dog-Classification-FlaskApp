import pandas as pd
import keras
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# from keras.preprocessing import image
import keras.utils as image

app = Flask(__name__)

def predict_label(image_path):
  #Input image
  test_image = image.load_img(image_path, target_size=(264,264))
  # test_image = image.load_img('/Users/akhilsharma/Datasets/dogs-vs-cats/train/cat.895.jpg', target_size=(264,264))
  # plt.imshow(test_image)
  test_image = image.img_to_array(test_image)
  test_image = np.expand_dims(test_image, axis=0)
  # Result array
  model = keras.models.load_model('model.h5')
  result = model.predict(test_image)
  
  #Mapping result array with the main name list
  i=0
  if(result>=0.5):
   return "Dog"
  else:
    return "Cat"


@app.route('/')
def main():
  return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
  if request.method == 'POST':
    img = request.files['my_image']
    
    img_path = 'static/' + img.filename
    img.save(img_path)
    
    p = predict_label(img_path)
    return render_template('index.html', prediction=p, img_path=img_path)


if __name__ == '__main__':
  app.run(port=8000, debug=True)