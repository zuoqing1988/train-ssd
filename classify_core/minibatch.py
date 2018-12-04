import cv2
import threading
from tools import image_processing
import numpy as np
import math
import sys,os
sys.path.append(os.getcwd())
from classify_config import config

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.ims, self.labels = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.ims, self.labels
        except Exception:
            return None

def get_minibatch_thread(imdb, mode = 'rgb'):
    num_images = len(imdb)
    processed_ims = list()
    cls_label = list()
    #print(num_images)
    for i in range(num_images):
        filename = config.root+'/classify_data/'+imdb[i]['image']
        #print(filename)
        im = cv2.imread(filename)
        h, w, c = im.shape
        cls = imdb[i]['label']
        cls_label.append(cls)
        if mode == 'rgb':
            im = im[:, :, ::-1]

        if imdb[i]['flipped']:
            im = im[:, ::-1, :]

        im_tensor = image_processing.transform(im)
        processed_ims.append(im_tensor)



    return processed_ims, cls_label

def get_minibatch(imdb, thread_num = 4, mode = 'bgr'):
    num_images = len(imdb)
    thread_num = max(2,thread_num)
    num_per_thread = math.ceil(float(num_images)/thread_num)
    #print(num_per_thread)
    threads = []
    for t in range(thread_num):
        start_idx = int(num_per_thread*t)
        end_idx = int(min(num_per_thread*(t+1),num_images))
        cur_imdb = [imdb[i] for i in range(start_idx, end_idx)]
        cur_thread = MyThread(get_minibatch_thread,(cur_imdb, mode))
        threads.append(cur_thread)
    for t in range(thread_num):
        threads[t].start()

    processed_ims = list()
    cls_label = list()
    
    for t in range(thread_num):
        cur_process_ims, cur_cls_label = threads[t].get_result()
        processed_ims = processed_ims + cur_process_ims
        cls_label = cls_label + cur_cls_label
        
    im_array = np.vstack(processed_ims)
    label_array = np.array(cls_label)
    
    data = {'data': im_array}
    label = {}
    label['label'] = label_array
    
    return data, label

def get_testbatch(imdb):
    assert len(imdb) == 1, "Single batch only"
    filename = config.root+'/classify_data/'+imdb[0]['image']
    #print(filename)
    im = cv2.imread(filename)
    cls = imdb[i]['label']
    cls_label.append(cls)
    if mode == 'rgb':
        im = im[:, :, ::-1]
    im_array = im
    data = {'data': im_array}
    label = {}
    return data, label
