from urllib import response
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
import urllib.request
import webbrowser

from threading import Thread




imread = imageio.imread


# Initialize the app

project_root = os.path.dirname(__file__)
template_path = os.path.join(project_root, './static/')

app = Flask(__name__, template_folder=template_path)

def get_model():
	global emotion_model, model,face_haar_cascade
	emotion_model = load_model('./model/model.h5')
	model = load_model('./model/bmodel.h5')


	face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
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

	objects = [ 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
	emotion = objects[result[0][0]]

	response = {
		'predicted_class' : emotion		 
	}
	# get_url= urllib.request.urlopen('https://www.google.com/')
	# print("Response Status: "+ str(get_url.getcode()) )
	# get_url= webbrowser.open('https://www.google.com/')
	return render_template('predict.html',data = emotion)

def funct(emotion):
	return render_template('predict.html',data = emotion)

# recommendation route
@app.route('/video', methods=['POST','GET'])

def video():
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
        count = count+1
        cv2.imwrite("frame%d.jpg"%count, test_img) 
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            img = image.load_img("./frame%d.jpg"%count, color_mode="grayscale", target_size=(48,48))
            img_pixels = image.img_to_array(img)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)

            # find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ( 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            t = Thread(target=funct, args=(predicted_emotion,))
            t.start()

			
            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ', resized_img)
        os.remove("./frame%d.jpg"%count)
        if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
            break

    cap.release()
    cv2.destroyAllWindows
    return render_template('predict.html',data = " ")



# Default app route
@app.route('/')
def hello():
	return render_template('predict.html', data = " ")

@app.route('/hello', methods = ['POST'])
def hi():
	message = request.get_json(force=True)
	name = message['name']
	response = {
		'greeting' : 'Hello, ' + name + '!'
	}
	#return jsonify(response)

if __name__ == '__main__':
    app.run()




	 

