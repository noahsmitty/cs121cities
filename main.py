"""
AUTHORS: Josh Cheung, Noah Smith, Arun Ramakrishna, Chuksi Emuwa, Giovanni Castro
DESCRIPTION: Our main app file using Flask and Fastai, responsible for taking an
             input image from our frontend, sending it to the appropriate models,
             returning a prediction, and routing between our two webpages.
"""
import os
from os import path
import sys
import pickle
import urllib.request
from fastai.learner import load_learner
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import current_app, send_from_directory
from werkzeug.utils import secure_filename



# set image directory for welcome page
UPLOAD_FOLDER = '/tmp/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# set allowed file types
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    """
    Sets the allowed file types upon image upload.
    :param filename is the filename of the input image.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def display_text(filename):
    """
    Uses Fastai to call our models and return a prediction, using an initial model
    to determine image category, and then feeding the image to the appropriate model.
    :param filename is the filename of the input image.
    :return an fstring of the prediction, displayed using WTForms on our HTML page.
    """
    model_dir = "models/"
    # run initial landmark vs skyline model
    learn_inf_categorize = load_learner(model_dir+"categorizeCPU.pkl", cpu=True)
    category = learn_inf_categorize.predict(UPLOAD_FOLDER + filename)[0]

    # run our other two models depending on first model
    if category == "landmarks":
        learn_inf = load_learner(model_dir+"landmarkModelCPU.pkl", cpu=True)
        pred = learn_inf.predict(UPLOAD_FOLDER + filename)[0]
    else:
        learn_inf = load_learner(model_dir+"skylineModelCPU.pkl", cpu=True)
        pred = learn_inf.predict(UPLOAD_FOLDER + filename)[0]

    # capitalize city names for display
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
    """
    Displays our welcome page.
    :return either our welcome.html page, or if button is clicked, go to upload_image.
    """
    if request.method == 'POST':
        return redirect(url_for('upload_image'))
    return render_template('welcome.html')


@app.route('/predict')
def upload_form():
    """
    Renders our main prediction page, with route /predict.
    :return our upload.html page.
    """
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def upload_image():
    """
    Our main function responsible for image upload, and the clearing of our upload folder
    after a prediction is made.
    :return back to the upload.html page, or rerender the page to display the uploaded image.
    """
    path = UPLOAD_FOLDER
    file_list = os.listdir(path)

    # this code clears everything in our image folder before we upload a new one
    for f_n in file_list:
        if f_n[-3:] == "jpg" or f_n[-4:] == "jpeg" or f_n[-3:] == "png":
            try:
                os.remove(path+f_n)
            except:
                print(f"Error while deleting file {path+f_n}")

    # check if our file is in request
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    # saves our uploaded file into appropriate folder
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
    """
    Sends image for display back to our frontend, downloading the image.
    :param filename, the filename of our uploaded file.
    :return file sent from the image directory /tmp/
    """
    uploads = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'])
    return send_from_directory(directory=uploads, filename=filename)


if __name__ == "__main__":
    app.run(debug=True)
