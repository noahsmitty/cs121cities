import os
from os import path
import sys
from app import app
import pickle
from fastai.learner import load_learner
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def display_text(filename, imagetype):
    learn_inf_categorize = load_learner("categorize.pkl", map_location=torch.device("cpu"))
    category = learn_inf_categorize.predict('static/uploads/' + filename)[0]

    if category == "landmarks":
        learn_inf = load_learner("landmarkModel.pkl", map_location=torch.device("cpu"))
        pred = learn_inf.predict('static/uploads/' + filename)[0]
    else:
        learn_inf = load_learner("skylineModel.pkl", map_location=torch.device("cpu"))
        pred = learn_inf.predict('static/uploads/' + filename)[0]

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

    return f"Your city is: {pred}"

@app.route('/', methods=['POST', 'GET'])
def welcome():
	if request.method == 'POST':
		return redirect(url_for('upload_form'))
	return render_template('welcome.html')


@app.route('/predict')
def upload_form():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        if "imagetype" not in request.values:
            flash("no image type selected")
            return redirect(request.url)

        imagetype = request.form["imagetype"]
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        predict = display_text(file.filename, imagetype)
        return render_template('upload.html', filename=filename, prediction = predict)
        
    flash('Allowed image types are -> png, jpg, jpeg')
    return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='static/uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)

