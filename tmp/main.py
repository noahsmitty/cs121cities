import os
from os import path
import sys
import pickle
from fastai.learner import load_learner
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, current_app, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/tmp/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def display_text(filename):
    learn_inf_categorize = load_learner("categorizeCPU.pkl", cpu=True)
    category = learn_inf_categorize.predict(UPLOAD_FOLDER + filename)[0]

    if category == "landmarks":
        learn_inf = load_learner("landmarkModelCPU.pkl", cpu=True)
        pred = learn_inf.predict(UPLOAD_FOLDER + filename)[0]
    else:
        learn_inf = load_learner("skylineModelCPU.pkl", cpu=True)
        pred = learn_inf.predict(UPLOAD_FOLDER + filename)[0]

    # make city names look nicer
    if pred == "san francisco":
        pred = "San Francisco"
    elif pred == "new york":
        pred = "New York"
    elif pred == "london":
        pred = "London"
    elif pred == "tokyo":
        pred = "Tokyo"
    else:
        pred = "New Delhi"

    return f"Prediction: {pred}"

@app.route('/', methods=['POST', 'GET'])
def welcome():
	if request.method == 'POST':
		return redirect(url_for('upload_image'))
	return render_template('welcome.html')


@app.route('/predict')
def upload_form():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def upload_image():
    path = UPLOAD_FOLDER
    fileList = os.listdir(path)
    
    for fn in fileList:
        if fn[-3:] == "jpg" or fn[-4:] == "jpeg" or fn[-3:] == "png":
            try:
                os.remove(path+fn)
            except:
                print(f"Error while deleting file {path+fn}")

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        full_fn = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        predict = display_text(file.filename)
        return render_template('upload.html', filename=filename, prediction=predict,\
                                image_fn=full_fn)
        
    flash('Allowed image types are -> png, jpg, jpeg')
    return redirect(request.url)

@app.route('/tmp/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    uploads = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'])
    return send_from_directory(directory=uploads, filename=filename)

"""
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for("static", "tmp/"+filename), code=301)
"""

if __name__ == "__main__":
    app.run(debug=True)

