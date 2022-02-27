from flask import Flask, render_template, request, jsonify


from imageio import imwrite
import imageio
import numpy as np
import keras.models
import re
import sys
import os
import cv2

# Dependencies
import base64
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import matplotlib.pyplot as plt


imread = imageio.imread


# Initialize the app

project_root = os.path.dirname(__file__)
template_path = os.path.join(project_root, './static/')

app = Flask(__name__, template_folder=template_path)

def get_model():
	global emotion_model
	emotion_model = load_model('./model/model.h5')
	# model._make_predict_function() # It gives TensorFlow error if this is omitted.
	print('Model Loaded!')

def preprocess_image(image, target_size):
	if image.mode != 'RGB':
		image = image.convert('RGB')
	image = image.resize(target_size)
	image = img_to_array(image)

	img = image.astype('float32')
	img /= 255
	c = np.zeros(32*32*3).reshape((1,32,32,3))
	c[0] = img
	return c

def emotion_analysis(emotions):
    objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, emotions, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('percentage')
    plt.title('emotion')
    
    plt.show()
	
print('Loading model..............') 
get_model()

# Prediction route
@app.route('/predict', methods=['POST','GET'])

def predict():
	# message = request.json(force=True)
	# print(message)
	# encoded = message['image']
	# decoded = base64.b64decode(encoded)
	# image = Image.open(io.BytesIO(decoded))
	# print(image)

	f = request.files['file']
	f.save(f.filename)
	file = "./"+f.filename

	# file = request.get_json(force=True)

	true_image = image.load_img(file)
	img = image.load_img(file, color_mode="grayscale", target_size=(48, 48))
	print("aaaa")
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis = 0)

	x /= 255

	custom = emotion_model.predict(x)
	print(custom[0])
	emotion_analysis(custom[0])

	x = np.array(x, 'float32')
	x = x.reshape([48, 48])


	plt.imshow(true_image)
	plt.show()

	result = np.where(custom[0]==max(custom[0]))
	print(result[0][0])

	objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
	emotion = objects[result[0][0]]

	response = {
		'predicted_class' : emotion
	}
	

	return jsonify(response)

# Default app route
@app.route('/')
def hello():
	return render_template('predict.html')

@app.route('/hello', methods = ['POST'])
def hi():
	message = request.get_json(force=True)
	name = message['name']
	response = {
		'greeting' : 'Hello, ' + name + '!'
	}
	return jsonify(response)

if __name__ == '__main__':
    app.run()