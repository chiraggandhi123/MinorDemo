from flask import Flask,render_template,url_for,request
import urllib
from PIL import Image
import requests
from io import BytesIO
import string
import os
import json
import siamese

model = load_model('./weights/siamese_model_resnet_4classes_new_weights.hd5')

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
app = Flask(__name__)


@app.route('/')
def home():
	return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		# print("HERE")
		message = request.form['myfile']
		data=(message)
		obj = siamese.classify(data,df_X_train,image_shape)
		print(classify.score(model))
		#print(data)
		print(message)
		path_image='./Image.jpg'
		image_full_path='/home/chirag/Desktop/PROJECT/MINOR'+path_image
		final_caption = "COVID"
	return render_template('result.html',prediction = final_caption,img_data=image_full_path)



if __name__ == '__main__':
	app.run(debug=True)
