import random
import numpy as np
import cv2
from functools import partial
import tensorflow as tf
import sys
import urllib.request
import os
import zipfile

output_image_counter = 0

def main(input_filename):
    #Step 1 - download google's pre-trained neural network (updated for Python 3)
    url = 'https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'
    data_dir = 'data/'
    model_name = os.path.split(url)[-1]

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    local_zip_file = os.path.join(data_dir, model_name)
    model_fn = 'tensorflow_inception_graph.pb'
    
    if not os.path.exists(os.path.join(data_dir, model_fn)):
        if not os.path.exists(local_zip_file):
            print("Downloading Inception model...")
            urllib.request.urlretrieve(url, local_zip_file)
        # Extract
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    # start with a gray image with a little noise
    img_noise = np.random.uniform(size=(224,224,3)) + 100.0

    #Step 2 - Creating Tensorflow session and loading the model
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)
    with tf.gfile.GFile(os.path.join(data_dir, model_fn), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf.placeholder(np.float32, name='input') # define the input tensor
    imagenet_mean = 117.0
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
    tf.import_graph_def(graph_def, {'input':t_preprocessed})

    layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

    print('Number of layers', len(layers))
    print('Total number of feature channels:', sum(feature_nums))

    def strip_consts(graph_def, max_const_size=32):
        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add()
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = b"<stripped %d bytes>"%size
        return strip_def

    def rename_nodes(graph_def, rename_func):
        res_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = res_def.node.add()
            n.MergeFrom(n0)
            n.name = rename_func(n.name)
            for i, s in enumerate(n.input):
                n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
        return res_def

    def visstd(a, s=0.1):
        return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

    def T(layer):
        return graph.get_tensor_by_name("import/%s:0"%layer)

    def tffunc(*argtypes):
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
            return wrapper
        return wrap

    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0,:,:,:]
    resize = tffunc(np.float32, np.int32)(resize)

    def calc_grad_tiled(img, t_grad, tile_size=512):
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                g = sess.run(t_grad, {t_input:sub})
                grad[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    def render_deepdream(t_obj, img0=img_noise, iter_n=10, step=1.8, octave_n=6, octave_scale=1.2):
        t_score = tf.reduce_mean(t_obj)
        t_grad = tf.gradients(t_score, t_input)[0]

        img = img0
        octaves = []
        for _ in range(octave_n-1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw)/octave_scale))
            hi = img-resize(lo, hw)
            img = lo
            octaves.append(hi)

        for octave in range(octave_n):
            if octave>0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2])+hi
            for _ in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                img += g*(step / (np.abs(g).mean()+1e-7))

        output_frame = img / 255.0
        output_frame = np.uint8(np.clip(output_frame, 0, 1)*255)
        return output_frame

    # Video processing
    cap = cv2.VideoCapture(input_filename)
    if not cap.isOpened():
        print(f"Cannot open {input_filename}, trying image mode...")
        img = cv2.imread(input_filename)
        if img is None:
            img = cv2.imread('items.png')
        output_frame = render_deepdream(tf.square(T('mixed3a')), img)
        cv2.imwrite('dreamed_image.jpg', output_frame)
        print("Image deep dream complete: dreamed_image.jpg")
        return

    writer = None
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = render_deepdream(tf.square(T('mixed3a')), frame)
        if writer is None:
            frame_size = (output_frame.shape[1], output_frame.shape[0])
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter('dreamed_video.mp4', fourcc, 30.0, frame_size)

        writer.write(output_frame)
        i += 1
        print(f'frame {i} complete.')

    cap.release()
    if writer:
        writer.release()
    print("Video deep dream complete: dreamed_video.mp4")

if __name__ == '__main__':
    input_filename = sys.argv[1] if len(sys.argv) > 1 else 'items.png'
    main(input_filename)

