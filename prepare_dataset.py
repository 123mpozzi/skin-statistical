import os, re, json, sys
import cv2
import imghdr
from random import shuffle
from utils import csv_sep, get_training_and_testing_sets


## USAGE: python prepare_dataset.py <name of the dataset (Schmugge, ECU, HGR)>

# Get the variable part of a filename into a dataset
def get_variable_filename(filename: str, format: str) -> str:
    if format == '':
        return filename

    # re.fullmatch(r'^img(.*)$', 'imgED (1)').group(1)
    # re.fullmatch(r'^(.*)-m$', 'att-massu.jpg-m').group(1)
    match =  re.fullmatch('^{}$'.format(format), filename)
    if match:
        return match.group(1)
    else:
        #print('Cannot match {} with pattern {}'.format(filename, format))
        return None

# only if exists! maybe return a warning
# args eg: datasets/ECU/skin_masks datasets/ECU/original_images datasets/ecu/
# note: NonDefined, TRain, TEst, VAlidation
def analyze_dataset(gt: str, ori: str, root_dir: str, note: str = 'nd',
                    gt_filename_format: str = '', ori_filename_format: str = '',
                    gt_ext: str = '', ori_ext: str = '') -> None:
    out_analysis_filename = 'data.csv'

    out_analysis = os.path.join(root_dir, out_analysis_filename)
    analyze_content(gt, ori, out_analysis, note = note,
                    gt_filename_format = gt_filename_format,
                    ori_filename_format = ori_filename_format,
                    gt_ext = gt_ext, ori_ext = ori_ext)

# creates a file with lines like: origina_image1.jpg, skin_mask1.png
def analyze_content(gt: str, ori: str, outfile: str, note: str = 'nd',
                    gt_filename_format: str = '', ori_filename_format: str = '',
                    gt_ext: str = '', ori_ext: str = '') -> None:
    # images found
    i = 0

    # append to data file
    with open(outfile, 'a') as out:

        for gt_file in os.listdir(gt):
            gt_path = os.path.join(gt, gt_file)

            # controlla se e' un'immagine (per evitare problemi con files come thumbs.db)
            if not os.path.isdir(gt_path) and imghdr.what(gt_path) != None:
                matched = False
                gt_name, gt_e = os.path.splitext(gt_file)
                gt_identifier = get_variable_filename(gt_name, gt_filename_format)

                if gt_identifier == None:
                    continue

                if gt_ext and gt_e != '.' + gt_ext:
                    continue
                
                for ori_file in os.listdir(ori):
                    ori_path = os.path.join(ori, ori_file)
                    ori_name, ori_e = os.path.splitext(ori_file)
                    ori_identifier = get_variable_filename(ori_name, ori_filename_format)
                    
                    if ori_identifier == None:
                        continue

                    if ori_ext and ori_e != '.' + ori_ext:
                        continue
                    
                    # try to find a match (original image - gt)
                    if gt_identifier == ori_identifier:
                        out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}\n")
                        i += 1
                        matched = True
                        break
                
                if not matched:
                    print(f'No matches found for {gt_identifier}')
            else:
                print(f'File {gt_path} is not an image')
        
        print(f"Found {i} images")

# (JSON) "processpipe" : "png,skin=255_255_255,invert"
#        "processout" : "out/process/folder" 
# skin=... vuol dire che la regola per binarizzare è: tutto quello che non è skin va a nero, skin a bianco
# bg=... viceversa
#    (quindi skin e bg fanno anche binarizzazione!)
# *il processing viene fatto nell'ordine scritto!!!
# 
# **in Schmugge da Uchile il bg ha colore bianco, da altro dataset ha un altro colore,
# nons erve permettere la duplicazione di parametri nella stringa process, basta definirli nei due file di import diversi
# si può fare anche per img normali, non masks (Schmugge)
def process_images(data_dir: str, process_pipeline: str, out_dir = '',
                   im_filename_format: str = '', im_ext: str = '') -> str:

    # loop mask files
    for im_basename in os.listdir(data_dir):
        im_path = os.path.join(data_dir, im_basename)
        im_filename, im_e = os.path.splitext(im_basename)

        # controlla se e' un'immagine (per evitare problemi con files come thumbs.db)
        if not os.path.isdir(im_path) and imghdr.what(im_path) != None:
            if out_dir == '':
                out_dir = os.path.join(data_dir, 'processed')

            os.makedirs(out_dir, exist_ok=True)

            im_identifier = get_variable_filename(im_filename, im_filename_format)
            if im_identifier == None:
                continue
            
            if im_ext and im_e != '.' + im_ext:
                continue

            # load image
            im = cv2.imread(im_path)

            # prepare path for out image
            im_path = os.path.join(out_dir, im_basename)

            for operation in process_pipeline.split(','):
                # binarize. Rule: what isn't skin is black
                if operation.startswith('skin'):
                    # inspired from https://stackoverflow.com/a/53989391
                    bgr_data = operation.split('=')[1]
                    bgr_chs = bgr_data.split('_')
                    b = int(bgr_chs[0])
                    g = int(bgr_chs[1])
                    r = int(bgr_chs[2])
                    lower_val = (b, g, r)
                    upper_val = lower_val
                    # Threshold the image to get only selected colors
                    # what isn't skin is black
                    mask = cv2.inRange(im, lower_val, upper_val)
                    im = mask
                # binarize. Rule: what isn't bg is white
                elif operation.startswith('bg'):
                    bgr_data = operation.split('=')[1]
                    bgr_chs = bgr_data.split('_')
                    b = int(bgr_chs[0])
                    g = int(bgr_chs[1])
                    r = int(bgr_chs[2])
                    lower_val = (b, g, r)
                    upper_val = lower_val
                    # Threshold the image to get only selected colors
                    mask = cv2.inRange(im, lower_val, upper_val)
                    #cv2_imshow(mask) #debug
                    # what isn't bg is white
                    sk = cv2.bitwise_not(mask)
                    im = sk
                # invert image
                elif operation == 'invert':
                    im = cv2.bitwise_not(im)
                # convert to png
                elif operation == 'png':
                    im_path = os.path.join(out_dir, im_filename + '.png')
                # reload image
                elif operation == 'reload':
                    im = cv2.imread(im_path)
                else:
                    print(f'Image processing operation unknown: {operation}')

            # save processing 
            cv2.imwrite(im_path, im)

    return out_dir

# import dataset and generate metadata
def import_dataset(import_json: str) -> None:
    if os.path.exists(import_json):
        with open(import_json, 'r') as stream:
            data = json.load(stream)

            # load JSON values
            gt = data['gt']
            ori = data['ori']
            root = data['root']
            note = data['note']
            gt_format = data['gtf']
            ori_format = data['orif']
            gt_ext = data['gtext']
            ori_ext = data['oriext']
            ori_process = data['oriprocess']
            ori_process_out = data['oriprocessout']
            gt_process = data['gtprocess']
            gt_process_out = data['gtprocessout']
            
            # check if processing is required
            if ori_process:
                ori = process_images(ori, ori_process, ori_process_out,
                                     ori_format, ori_ext)
                # update the file extension in the images are being converted
                if 'png' in ori_process:
                    ori_ext = 'png'
            
            if gt_process:
                gt = process_images(gt, gt_process, gt_process_out,
                                     gt_format, gt_ext)
                if 'png' in gt_process:
                    gt_ext = 'png'
            
            # Non-Defined as default note
            if not note:
                note = 'nd'
            
            # analyze the dataset and create the csv files
            analyze_dataset(gt, ori, root,
                            note, gt_format, ori_format,
                            gt_ext, ori_ext)
    else:
        print("JSON import file does not exist!")

# randomize splits inside a data.csv file
def process_random(csv_file: str, outfile: str, ori_out_dir = 'new_ori', gt_out_dir = 'new_gt'):
    # prepare new ori and gt dirs
    os.makedirs(ori_out_dir, exist_ok=True)
    os.makedirs(gt_out_dir, exist_ok=True)

    file = open(csv_file)
    file3c = file.read().splitlines()
    file.close()

    with open(outfile, 'w') as out:
        shuffle(file3c) # randomize

        # 70% train, 15% val, 15% test
        train_files, test_files = get_training_and_testing_sets(file3c)
        test_files, val_files = get_training_and_testing_sets(test_files, split=.5)

        for entry in file3c:
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]

            ori_path = os.path.normpath(ori_path)
            gt_path = os.path.normpath(gt_path)

            skintone = ''
            if len(entry.split(csv_sep)) == 4:
                skintone = csv_sep + entry.split(csv_sep)[3]
            
            note = 'te'
            if entry in train_files:
                note = 'tr'
            elif entry in val_files:
                note = 'va'
            
            out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{csv_sep}{skintone}\n")


# generate datasets metadata
if __name__ == "__main__":
    # total arguments
    n = len(sys.argv)

    if n != 2:
        exit('There must be 1 argument, that is the dataset name (Schmugge, ECU, HGR)')

    dataset = sys.argv[1] # first argument, argv[0] is the name of the script
    #dataset = 'Schmugge'
    dataset = dataset.upper()

    if dataset == 'ECU':
        import_dataset("dataset/import_ecu.json")
    elif dataset == 'HGR' or dataset == 'HGR_small':
        # hgr is composed of 3 sub datasets
        import_dataset("dataset/import_hgr1.json")
        import_dataset("dataset/import_hgr2a.json")
        import_dataset("dataset/import_hgr2b.json")
        process_random('dataset/HGR_small/data.csv', 'dataset/HGR_small/data.csv')
    elif dataset == 'SCHMUGGE':
        # schmugge dataset has really different filename formats but has a custom config file included
        schm = read_schmugge('dataset/Schmugge/data/.config.SkinImManager', 'dataset/Schmugge/data/data')
        # load a specific schmugge data.csv (use process_schmugge to create new splits)
        import_schmugge(schm, 'dataset/Schmugge/data.csv', 'dataset/Schmugge/schmugge_import.csv', ori_out_dir='dataset/Schmugge/newdata/ori', gt_out_dir='dataset/Schmugge/newdata/gt')
    else:
        exit('Invalid dataset!')