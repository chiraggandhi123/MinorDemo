from flask import Flask,render_template,url_for,request
import urllib
from PIL import Image
import requests
from io import BytesIO
import string
import os
import json
import pandas as pd
from keras.models import Model, load_model
import siamese
model = load_model('weights/siamese_model_resnet_4classes_new_weights.hd5')

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
app = Flask(__name__)

@app.before_first_request
def initialize():
    print ("Called only once, when the first request comes in")
@app.route('/')
def home():
	return render_template('home.html')
@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		# print("HERE")
		message = request.form['myfile']
		data = siamese.load_image('./static/test_image/'+ message)
		# print(message)
		path = './static/test_image/'+message
		print(path)
		new_data = [['chirag',path]]
		df = pd.DataFrame(new_data,columns = ['Class','Path'])
		# df.append('chirag',path)
		obj = siamese.Classify(df,siamese.df_X_train,siamese.image_shape)
		img = obj.resize_image(data)

		ans = obj.score(model)
		img_data_path = 'test_image/'+message
		# print(obj.score(model))
		# print(classify.score(model))
		#print(data)
		img_url = url_for('static',filename = img_data_path )
	return render_template('result.html',prediction = ans[1][0],img_data=img_url)


if __name__ == '__main__':
	
	app.run(port=4004, debug=True, host='0.0.0.0', use_reloader=False)


