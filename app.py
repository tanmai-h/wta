import os
import uuid
from flask import Flask, render_template, redirect, url_for, request, flash
from werkzeug.utils import secure_filename as sfn
from model import model as md
from model import tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

ALLOWED_EXTS = set(['.jpg', '.jpeg', '.png'])
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
PORT = 8888
INPUT_SHAPE = (192,256)

app = Flask(__name__)
md.load_weights("model.h5")
global graph
graph = tf.get_default_graph()

# App configuration
def configure(app):
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['RESULT_FOLDER'] = RESULT_FOLDER
    app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        # Error!
        return 'file not sent'
    file = request.files['file']
    if not file.filename:
        # Error!
        return 'filename error'
    _, file_ext = os.path.splitext(file.filename)
    if file_ext not in ALLOWED_EXTS:
        # Error!
        return 'file ext not allowed'
    filename = uuid.uuid4().hex + file_ext
    file.save(os.path.join('static', app.config['UPLOAD_FOLDER'], filename))
    return redirect(url_for('result', uploaded_img=filename))

@app.route('/result', methods=['GET'])
def result():
    uploaded_img = os.path.join('static', app.config['UPLOAD_FOLDER'], request.args['uploaded_img'])
    result_img = os.path.join('static', app.config['RESULT_FOLDER'], request.args['uploaded_img'])
    img_obj = Image.open(uploaded_img).resize(INPUT_SHAPE)
    img = np.asarray(img_obj)
    img = np.transpose(img, (1,0,2))
    # print(img.shape)
    app.logger.debug(img.shape)
    fin_img = np.expand_dims(img, 0)
    with graph.as_default():
        ret_img = md.predict(fin_img)[0].T
        plt.imsave(result_img, ret_img, cmap='gray')
        app.logger.debug(ret_img.shape)
    return render_template('result.html', 
                            uploaded_img=uploaded_img, 
                            result_img=result_img)

if __name__ == "__main__":
    configure(app)
    app.run(port=PORT, debug=True)