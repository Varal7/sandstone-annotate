from flask import Flask, send_file, render_template, request
from PIL import Image
import flask
import numpy as np
import os
import json
from collections import defaultdict
import random
import io
from string import ascii_uppercase

app = Flask(__name__)

NUM_CLASSES = 2
NUM_IMAGES = 3
NUM_CHANNELS = 5

METADATA_DIR = '/data/rsg/mammogram/CellPainter'
metadata_file = "images_for_puma_and_morpho_assay_metadata_aug18_2020.json"
full_dataset_dir = '/data/rsg/mammogram/CellPainter/ml.jku.at/software/cellpainting/dataset/full_dataset/'

metadata = json.load(open(os.path.join(METADATA_DIR, metadata_file)))

def get_random_smile():
    return random.choices(list(metadata.keys()))[0]

def get_random_image_basenames(smile, k=3):
    images = metadata[smile]['images']
    plate2images = defaultdict(list)
    for x in images:
        plate2images[x['plate_id']].append(os.path.basename(x['path']))

    plates = plate2images.keys()
    k = min(k, len(plates))
    chosen_plates = np.random.choice(list(plates), size=k, replace=False)

    return [np.random.choice(plate2images[p], size=1)[0] for p in chosen_plates]

@app.route('/')
def main():
    mols = []
    for i in range(NUM_CLASSES):
        smile = get_random_smile()
        image_basenames = get_random_image_basenames(smile, NUM_IMAGES + 1)
        mols.append({
            'smile': smile,
            'shortname': ascii_uppercase[i],
            'image_basenames': image_basenames[:-1],
            'test_image': image_basenames[-1]
        })

    test_index = random.choices(range(NUM_CLASSES))[0]

    last = request.args.get('last', 'NA')

    return render_template("main.html", mols=mols, test_index=test_index, num_channels=NUM_CHANNELS, last=last)

@app.route('/random')
def debug():
    smile = get_random_smile()
    image_basenames = get_random_image_basenames(smile)
    text =  'Random molecule: {}\n'.format(smile)
    text +=  'basenames: {}'.format(image_basenames)
    return text

@app.route('/image/<basename>/<int:dim>')
def image(basename, dim):
    if dim < 0 or dim >= 5:
        print("abort")
        flask.abort(404)
    img_io = io.BytesIO()
    path = os.path.join(full_dataset_dir, basename + ".npz")
    data = np.load(path)
    arr = data['sample']
    img = Image.fromarray(arr[:,:,dim])
    img.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/PNG')

