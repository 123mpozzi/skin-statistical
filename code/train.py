import csv
import os
import numpy as np
from tqdm import tqdm

from predict import open_image
from utils.logmanager import *

# FUTURE improvement ideas
# -use sparse matrix for saving model
# -use numpy to avoid the 3d histogram nested loops in data()
# -use cv2 to read images, should be faster


## this function reads image and get RGB data
def read_image(path: str):
    '''Return a sequential list of R,G,B values representing an image'''
    im = open_image(path)
    im_data = im.getdata()
    im.close()
    return list(im_data) # listing all rgb into a list

def is_skin(rgb, threshold: int = 150):
    '''Grountruth pixel is skin if it is whiteish'''
    r, g, b = rgb
    return r > threshold and g > threshold and b > threshold

def train_data(im_data, y_data, skin, non_skin):
    '''Add more data to the 3-dimensional histograms'''
    for i in range(len(im_data)):
        r, g, b = im_data[i]

        if is_skin(y_data[i]):
            # incrementing skin value(default = 0) of current rgb combination
            skin[r][g][b] += 1
        else:
            # incrementing non_skin value(default = 0) of current rgb combination
            non_skin[r][g][b] += 1
       
    return skin, non_skin
 
def calc_probability(skin, non_skin):
    '''Probability function'''
    # ex: probability[10][20][30] = skin[10][20][30]/(skin[10][20][30] + non_skin[10][20][30])
    return list(skin / (non_skin + skin))

def data(probability):  ## just a function to make list of rgb and prob
    '''Return a list of tuples (R,G,B,p) where p is the probability assigned to the RGB triplet'''
    arr = []
    
    for r in tqdm(range(256)):
        for g in range(256):
            for b in range(256):
                arr.append((r, g, b, probability[r][g][b])) # Nan on CSV if probability is None
    
    return arr

def create_csv(probability, filename):
    '''Create model CSV file'''
    myFile = open(filename, 'w', newline = '')
    with myFile:  
        writer = csv.writer(myFile)
        writer.writerow(["Red", "Green", "Blue", "Probability"])
        writer.writerows(data(probability))
    info('Training Completed')

def do_training(image_paths, out):
    # 3D histograms which represent training data
    skin = np.zeros((256,256,256)) 
    non_skin = np.zeros((256,256,256))    
    
    info('Reading training images...')
    for i in tqdm(image_paths):
        im_abspath = os.path.abspath(i[0])
        y_abspath = os.path.abspath(i[1])

        im = read_image(im_abspath) # storing the pixels of actual picture..
        y = read_image(y_abspath) # storing the pixels of mask picture..
        skin, non_skin = train_data(im, y, skin, non_skin)
    
    probability = calc_probability(skin, non_skin)

    info('Saving training data...')
    create_csv(probability, out) # creating CSV from that probabilty and rgb
