from flask import Flask, render_template, request, jsonify, flash, redirect
from layoutlm_inference import run_inference
from PIL import Image  
import os
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from flask_cors import CORS


app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    file = request.files['my_image']
    p, image = run_inference(file.read())
    print(p["company"])
    return render_template("index.html", prediction = p, img_path = file.filename, image = image, image_origin=file)

@app.route("/api/process", methods = ['POST'])
def process_img():
    file = request.files['file']
    print(file.filename)
    p, image = run_inference(file.read())
    data = {
      'total': p["total"],
      'date': p["date"],
      'company': p["company"],
      'address': p["address"]
    }
    return jsonify(data)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)