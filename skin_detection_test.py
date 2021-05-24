from prepare_dataset import process_schmugge, read_schmugge
from PIL import Image
import pandas as pd
from utils import csv_note_count, csv_skintone_count, csv_skintone_filter, get_test_paths, get_all_paths, load_skintone_split, get_bench_testset
from tqdm import tqdm
import sys, os, time
from shutil import copyfile


def open_image(src): 
    return Image.open(src,'r')

# measure_time is the file to open and append performance data
def predict(probability, path_x, path_y, out_dir, measure_time: str = None):
    im = open_image(path_x)
    temp = im.copy()
    # use the x filename for all saved images filenames (x, y, p)
    filename, x_ext = os.path.splitext(os.path.basename(path_x))
    # the masks and predictions will be saved LOSSLESS as PNG
    out_p = f'{out_dir}/p/{filename}.png'
    out_y = f'{out_dir}/y/{filename}.png'
    out_x = f'{out_dir}/x/{filename}{x_ext}'

    # save p
    t_elapsed = create_image(temp, probability, out_p)
    #t_elapsed = create_image_t(temp, probability, out_p)

    # copy x and y
    copyfile(path_x, out_x)
    copyfile(path_y, out_y)

    #temp.close()
    #im.close()

    # save performance
    if measure_time:
        with open(measure_time, 'a') as myfile:
            myfile.write(f'{path_x},{t_elapsed}\n')

def create_image(im, probability, out_p): 
    width, height = im.size
    pix = im.load()

    t_start = time.time()
    # ALGO
    for i in range(width):
        for j in range(height):
            r,g,b = im.getpixel((i,j))
            row_num = (r*256*256) + (g*256) + b #calculating the serial row number 
            if(probability['Probability'][row_num] <0.555555):
                pix[i,j] = (0,0,0)
            else:
                pix[i,j] = (255,255,255)
    t_elapsed = time.time() - t_start
    
    im.save(out_p)
    return t_elapsed

# from https://stackoverflow.com/a/36469395
# experimental, maybe faster ? NOT TESTED
def create_image_t(im, probability, out_p): 
    im.load()

    t_start = time.time()
    # ALGO
    newimdata = []
    #redcolor = (255,0,0)
    whitecolor = (255,255,255)
    blackcolor = (0,0,0)
    #for color in im.getdata():
    for r,g,b in im.getdata():
        #r = color[0]
        #g = color[1]
        #b = color[2]
        row_num = (r*256*256) + (g*256) + b #calculating the serial row number 
        if(probability['Probability'][row_num] <0.555555):
        #if color == redcolor:
            newimdata.append( whitecolor )
        else:
            newimdata.append( blackcolor )
    
    newim = Image.new(im.mode,im.size)
    newim.putdata(newimdata)
    t_elapsed = time.time() - t_start
    
    im.save(out_p)
    return t_elapsed

def main(image_paths, in_model, out_dir, bench_outfile: str = None):
    print("Reading CSV...")
    probability = pd.read_csv(in_model) # getting the rows from csv    
    print('Data collection completed')

    # make dirs
    os.makedirs(f'{out_dir}/p', exist_ok=True)
    os.makedirs(f'{out_dir}/y', exist_ok=True)
    os.makedirs(f'{out_dir}/x', exist_ok=True)

    for i in tqdm(image_paths):
        im_abspath = os.path.abspath(i[0]) 
        y_abspath = os.path.abspath(i[1])

        predict(probability, im_abspath, y_abspath, out_dir, measure_time=bench_outfile) # this tests the data

def get_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


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
    

    ## ALL: predict using all the default datasets
    if db_model == 'all':
        timestr = get_timestamp()
        modls = ['ecu', 'hgr', 'schmugge']
        datas = ['./dataset/HGR_small', './dataset/ECU', './dataset/Schmugge']


        # base preds: based on splits defined by me and only predict on self
        # timestr/bayes/base/{ecu,hgr,schmugge}/{p/y/x}
        for in_model in modls: # load each model
            inm = in_model

            if in_model == 'ecu':
                in_model = 'ECU'
            elif in_model == 'hgr':
                in_model = 'HGR_small'
            elif in_model == 'schmugge':
                in_model = 'Schmugge'

            model_name = in_model + '.csv'
            
            in_dir = f'./dataset/{in_model}'
            out_dir = f'./predictions/{timestr}/bayes/base/{inm}'

            image_paths = get_test_paths(os.path.join(in_dir, 'data.csv'))
            main(image_paths, model_name, out_dir)


        # cross preds: use a dataset whole as the testing set
        # timestr/bayes/cross/ecu_on_ecu/{p/y/x}
        for in_model in modls: # load each model
            inm = in_model

            if in_model == 'ecu':
                in_model = 'ECU'
            elif in_model == 'hgr':
                in_model = 'HGR_small'
            elif in_model == 'schmugge':
                in_model = 'Schmugge'
            
            model_name = in_model + '.csv'

            for ds in datas:
                inp = os.path.basename(ds).lower()
                #in_dir = f'./dataset/{in_model}'
                out_dir = f'./predictions/{timestr}/bayes/cross/{inm}_on_{inp}'

                # use whole dataset as testing set
                image_paths = get_all_paths(os.path.join(ds, 'data.csv'))
                main(image_paths, model_name, out_dir)
    

    ## SKINTONES: predict all on Schmugge skintones
    elif db_model == 'skintones':
        timestr = get_timestamp()
        modls = ['dark', 'medium', 'light']
        in_dir = './dataset/Schmugge'


        # base preds: based on splits defined by me and only predict on self
        # timestr/bayes/base/{ecu,hgr,schmugge}/{p/y/x}
        for in_model in modls: # load each model
            model_name = in_model + '.csv'

            load_skintone_split(in_model)
            
            out_dir = f'./predictions/{timestr}/bayes/base/{in_model}'
            image_paths = get_test_paths(os.path.join(in_dir, 'data.csv'))

            main(image_paths, model_name, out_dir)


        # cross preds: use a dataset whole as the testing set
        # timestr/bayes/cross/ecu_on_ecu/{p/y/x}
        for in_model in modls: # load each model
            model_name = in_model + '.csv'

            for inp in modls:
                # re-import Schmugge
                schm = read_schmugge('./dataset/Schmugge/data/.config.SkinImManager', './dataset/Schmugge/data/data')
                process_schmugge(schm, './dataset/Schmugge/data.csv', ori_out_dir='./dataset/Schmugge/newdata/ori', gt_out_dir='./dataset/Schmugge/newdata/gt')

                # Set to True to filter a dataset CSV by given skintone,
                # the other entries will be deleted from the CSV file
                filter_by_skintone_csv = './dataset/Schmugge/data.csv' # dataset to process
                csv_skintone_filter(filter_by_skintone_csv, inp, mode = 'test')
                csv_skintone_count(filter_by_skintone_csv, inp)
                #csv_note_count(filter_by_skintone_csv, filter_mode)

                #in_dir = f'./dataset/{in_model}'
                out_dir = f'./predictions/{timestr}/bayes/cross/{in_model}_on_{inp}'

                # use whole dataset as testing set
                image_paths = get_test_paths(os.path.join(in_dir, 'data.csv'))
                main(image_paths, model_name, out_dir)
    

    ## BENCHMARK: to measure the execution time
    elif db_model == 'bench':
        timestr = get_timestamp()

        # use first 15 ECU images as test set
        in_dir = './dataset/ECU'
        db_csv = os.path.join(in_dir, 'data.csv')
        in_model = 'ECU.csv'
        out_dir = f'./predictions/bench/{timestr}'

        # set only the first 15 ECU images as test
        get_bench_testset(db_csv, count=15)

        image_paths = get_test_paths(db_csv)

        # do 5 observations
        # the predictions will be the same
        # but the performance will be logged 5 times
        for k in range(5):
            main(image_paths, in_model, out_dir, f'bench{k}.txt')
    

    ## DEFAULT: normal case
    else: 
        # Create output dirs if not exist
        os.makedirs(out_p, exist_ok=True)
        os.makedirs(out_y, exist_ok=True)


        # special case: predict on Schmugge skintones sub-splits
        if db_pred in ('light', 'medium', 'dark'):
            # re-import Schmugge
            schm = read_schmugge('./dataset/Schmugge/data/.config.SkinImManager', './dataset/Schmugge/data/data')
            process_schmugge(schm, './dataset/Schmugge/data.csv', ori_out_dir='./dataset/Schmugge/newdata/ori', gt_out_dir='./dataset/Schmugge/newdata/gt')


            # Set to True to filter a dataset CSV by given skintone,
            # the other entries will be deleted from the CSV file
            filter_by_skintone = True
            filter_by_skintone_type = db_pred # light medium dark nd
            filter_by_skintone_csv = './dataset/Schmugge/data.csv' # dataset to process
            filter_mode = 'test'

            if filter_by_skintone:
                csv_skintone_filter(filter_by_skintone_csv, filter_by_skintone_type, mode = filter_mode)
                csv_skintone_count(filter_by_skintone_csv, filter_by_skintone_type)
                csv_note_count(filter_by_skintone_csv, filter_mode)
            
            in_dir = './dataset/Schmugge'


        image_paths = get_test_paths(os.path.join(in_dir, 'data.csv'))
        main(image_paths, in_model, out_p, out_y)

