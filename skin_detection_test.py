from PIL import Image
import pandas as pd
from utils import get_test_paths
from tqdm import tqdm
import sys, os


def open_image(src): 
    return Image.open(src,'r')

                   
def test(probability, path_p, path_y, out_p, out_y):
    
    im = open_image(path_p)
    temp = im.copy()
    filename, p_ext = os.path.splitext(os.path.basename(path_p))
    create_image(temp, probability, os.path.join(out_p, filename + '.png'))

    im_y = open_image(path_y)
    # rename the mask using the prediction name, but as PNG
    im_y.save(os.path.join(out_y, filename + '.png'))


def create_image(im, probability, out_p): 
    
    width, height = im.size

    pix = im.load()
  
    for i in range(width):
        for j in range(height):
            r,g,b = im.getpixel((i,j))
            row_num = (r*256*256) + (g*256) + b #calculating the serial row number 
            if(probability['Probability'][row_num] <0.555555):
                pix[i,j] = (0,0,0)
            else:
                pix[i,j] = (255,255,255)
    
    im.save(out_p)
    #saveImage(im)
    
def saveImage(image): ## saving image
    image.save('test/result.jpg')

def main(image_paths, in_model, out_p, out_y):

    print("Reading CSV...")
    probability = pd.read_csv(in_model) # getting the rows from csv    
    print('Data collection completed') 
    
    #path = 'test/1.jpg'
    for i in tqdm(image_paths):
        im_abspath = os.path.abspath(i[0]) 
        y_abspath = os.path.abspath(i[1])

        test(probability, im_abspath, y_abspath, out_p, out_y) # this tests the data


    #print("Image created")


if __name__ == "__main__":
    # total arguments
    n = len(sys.argv)

    if n == 2: # predict over the same dataset of the model
        db_model = sys.argv[1] # first argument, argv[0] is the name of the script
        in_model = f'./{db_model}.csv'
        in_dir = f'./dataset/{db_model}'
        out_p = f'./predictions/{db_model}_on_{db_model}/p'
        out_y = f'./predictions/{db_model}_on_{db_model}/y'
    elif n == 3: # cross dataset predictions
        db_model = sys.argv[1] # first argument, argv[0] is the name of the script
        db_pred = sys.argv[2]
        in_model = f'./{db_model}.csv'
        in_dir = f'./dataset/{db_pred}'
        out_p = f'./predictions/{db_model}_on_{db_pred}/p'
        out_y = f'./predictions/{db_model}_on_{db_pred}/y'
    else:
        exit('''There must be 1 or 2 arguments!
        Usage: python skin_detection_test.py db-model [db-predict]
        db examples: Schmugge, ECU, HGR''')

    # Create output dirs if not exist
    os.makedirs(out_p, exist_ok=True)
    os.makedirs(out_y, exist_ok=True)

    image_paths = get_test_paths(os.path.join(in_dir, 'data.csv'))

    main(image_paths, in_model, out_p, out_y)

