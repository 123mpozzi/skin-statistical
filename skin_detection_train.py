import os, sys
from PIL import Image
import numpy as np
import csv
import pandas as pd
from utils import get_train_paths
from tqdm import tqdm


##src- path of image file
def open_image(src): 
    return Image.open(src,'r')


## this function reads image and get RGB data
def read_image(im):
    im = im.convert('RGB') ##converts single value to rgb
    return list(im.getdata()) ##listing all rgb into a list
  

def check_skin(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    if (r <= 150 and g <= 150 and b <= 150): return False
    return True

def train_data(pixels, pix_val_actual, pix_val_mask, skin, non_skin):
        
    for i in range(len(pix_val_actual)):
        
        r = pix_val_actual[i][0]
        g = pix_val_actual[i][1]
        b = pix_val_actual[i][2]

        if(check_skin(pix_val_mask[i])):
            skin[r][g][b] += 1 
      
        else:
            non_skin[r][g][b] += 1
       
    return pixels, skin, non_skin                  
 
def set_probability(pixel, skin, non_skin, probability):
    probability = list(skin / (non_skin + skin))
    return probability


def to_list(r,g,b,probability):
    a = []
    a.append(r)
    a.append(g)
    a.append(b)
    a.append(probability)
    return list(a)

def data(probability):  ## just a function to make list of rgb and prob
    arr = []
    
    #for r in range(256):
    for r in tqdm(range(256)):
        for g in range(256):
            for b in range(256):
                arr.append(to_list(r,g,b,probability[r][g][b]))
                progress += 1
        
    return arr     


def create_csv(probability, filename): ##this function creats csv 
    myFile = open(filename, 'w', newline = '')
    with myFile:  
        writer = csv.writer(myFile)
        writer.writerow(["Red", "Green", "Blue", "Probability"])
        writer.writerows(data(probability))
    print('Training Completed')


def main(image_paths, out):

    pixels = np.zeros((256,256,256))
    skin = np.zeros((256,256,256)) 
    non_skin = np.zeros((256,256,256))    
    probability = np.zeros((256,256,256))

    #files_actual = os.listdir('image')      #all filenames of that particular dir -- image
    #files_mask = os.listdir('mask')         #all filenames of that particular dir -- mask
    
    print('Reading training files...')

    #for i in range(len(files_actual)): ##iterating through all images
    for i in tqdm(image_paths):
        im_abspath = os.path.abspath(i[0]) 
        y_abspath = os.path.abspath(i[1])
       
        #image_actual_path = 'image\\'
        #image_mask_path = 'mask\\'

        #pix_val_actual = read_image(open_image(image_actual_path+files_actual[i])) ## storing the pixels of actual picture..
        #pix_val_mask = read_image(open_image(image_mask_path+files_mask[i])) ## storing the pixels of mask picture..
        pix_val_actual = read_image(open_image(im_abspath))
        pix_val_mask = read_image(open_image(y_abspath))

        #print(image_actual_path+files_actual[i], image_mask_path+files_mask[i])
        
        pixels, skin, non_skin = train_data(pixels, pix_val_actual, pix_val_mask, skin, non_skin) ## this returns the skin value and non_skin value 
#        
    probability = set_probability(pixels, skin, non_skin, probability) ## this returns the probability

    print('Saving training data...')
    create_csv(probability, out) ## creating CSV from that probabilty and rgb


## USAGE: python skin_detection_train.py <name of the dataset (Schmugge, ECU, HGR)>
if __name__ == "__main__":
    # total arguments
    n = len(sys.argv)

    if n != 2:
        exit('''There must be 1 argument!
        Usage: python skin_detection_train.py <db-name>
        db examples: Schmugge, ECU, HGR''')

    dataset = sys.argv[1]

    if dataset == 'HGR':
        dataset = 'HGR_small'
    elif dataset in ('light', 'medium', 'dark'):
        name_in = 'Schmugge'
        name_out = dataset

    in_dir = f'./dataset/{name_in}'
    out = f'./{name_out}.csv'
    #in_dir = f'./dataset/{dataset}'
    #out = f'./{dataset}.csv'

    image_paths = get_train_paths(os.path.join(in_dir, 'data.csv'))

    main(image_paths, out)