import os, re, json, sys
import cv2
import imghdr
from random import shuffle
from math import floor
from utils import csv_sep


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

# from schmugge custom config (.config.SkinImManager) to a list of dict structure
def read_schmugge(skin_im_manager_path: str, images_dir: str) -> list: # also prepare the csv
    sch = []
    
    # images with gt errors, aa69 is also duplicated in the config file
    blacklist = ['aa50.gt.d3.pgm', 'aa69.gt.d3.pgm', 'dd71.gt.d3.pgm', 'hh54.gt.d3.pgm']

    with open(skin_im_manager_path) as f:
        start = 0
        i = 0
        tmp = {}
        for line in f:
            blacklisted = False

            if start < 2: # skip first 2 lines
                start += 1
                continue
            
            if line: # line not empty
                line = line.rstrip() # remove End Of Line (\n)

                if i == 2: # skin tone type
                    skin_type = int(line)
                    if skin_type == 0:
                        tmp['skintone'] = 'light'
                    elif skin_type == 1:
                        tmp['skintone'] = 'medium'
                    elif skin_type == 2:
                        tmp['skintone'] = 'dark'
                    else:
                        tmp['skintone'] = 'nd'
                elif i == 3: # db type
                    tmp['db'] = line
                elif i == 8: # ori
                    tmp['ori'] = os.path.join(images_dir, line)
                elif i == 9: # gt
                    tmp['gt'] = os.path.join(images_dir, line)
                    if line in blacklist:
                        blacklisted = True
                

                # update image counter
                i += 1
                if i == 10: # 10 lines read, prepare for next image data
                    if not blacklisted:
                        sch.append(tmp)
                    tmp = {}
                    i = 0
    
    print(f'Schmugge custom config read correctly, found {len(sch)} images')

    return sch

# from schmugge list of dicts structure to csv file and processed images
def process_schmugge(sch: list, outfile: str, ori_out_dir = 'new_ori', gt_out_dir = 'new_gt'):
    # prepare new ori and gt dirs
    os.makedirs(ori_out_dir, exist_ok=True)
    os.makedirs(gt_out_dir, exist_ok=True)

    with open(outfile, 'w') as out:
        shuffle(sch) # randomize

        # 70% train, 15% val, 15% test
        train_files, test_files = get_training_and_testing_sets(sch)
        test_files, val_files = get_training_and_testing_sets(test_files, split=.5)

        for entry in sch:
            db = int(entry['db'])
            ori_path = entry['ori']
            gt_path = entry['gt']
            

            ori_basename = os.path.basename(ori_path)
            gt_basename = os.path.basename(gt_path)
            ori_filename, ori_e = os.path.splitext(ori_basename)
            gt_filename, gt_e = os.path.splitext(gt_basename)

            # process images
            # load images
            ori_im = cv2.imread(ori_path)
            gt_im = cv2.imread(gt_path)
            # png
            ori_out = os.path.join(ori_out_dir, ori_filename + '.png')
            gt_out = os.path.join(gt_out_dir, gt_filename + '.png')
            # on windows the path slashes are mixed
            ori_out = os.path.normpath(ori_out)
            gt_out = os.path.normpath(gt_out)
            # binarize gt: whatever isn't background, is skin
            if db == 4 or db == 3: # Uchile/UW: white background
                b = 255
                g = 255
                r = 255
                lower_val = (b, g, r)
                upper_val = lower_val
                # Threshold the image to get only selected colors
                mask = cv2.inRange(gt_im, lower_val, upper_val)
                # what isn't bg is white
                sk = cv2.bitwise_not(mask)
                gt_im = sk
            else: # background = 180,180,180
                b = 180
                g = 180
                r = 180
                lower_val = (b, g, r)
                upper_val = lower_val
                # Threshold the image to get only selected colors
                mask = cv2.inRange(gt_im, lower_val, upper_val)
                # what isn't bg is white
                sk = cv2.bitwise_not(mask)
                gt_im = sk
            # save processing 
            cv2.imwrite(ori_out, ori_im)
            cv2.imwrite(gt_out, gt_im)

            skintone = entry['skintone']
            note = 'te'
            if entry in train_files:
                note = 'tr'
            elif entry in val_files:
                note = 'va'
            
            out.write(f"{ori_out}{csv_sep}{gt_out}{csv_sep}{note}{csv_sep}{skintone}\n")

# from schmugge list of dicts structure and csv_file with pre-defined splits to csv file and processed images
def import_schmugge(sch: list, outfile: str, csv_file: str, ori_out_dir = 'new_ori', gt_out_dir = 'new_gt'):
    # prepare new ori and gt dirs
    os.makedirs(ori_out_dir, exist_ok=True)
    os.makedirs(gt_out_dir, exist_ok=True)

    file = open(csv_file)
    file3c = file.read().splitlines()
    file.close()

    with open(outfile, 'w') as out:
        for entry in sch:
            db = int(entry['db'])
            ori_path = entry['ori']
            gt_path = entry['gt']
            
            ori_basename = os.path.basename(ori_path)
            gt_basename = os.path.basename(gt_path)
            ori_filename, ori_e = os.path.splitext(ori_basename)
            gt_filename, gt_e = os.path.splitext(gt_basename)

            # process images
            # load images
            ori_im = cv2.imread(ori_path)
            gt_im = cv2.imread(gt_path)
            # png
            ori_out = os.path.join(ori_out_dir, ori_filename + '.png')
            gt_out = os.path.join(gt_out_dir, gt_filename + '.png')
            # on windows the path slashes are mixed
            ori_out = os.path.normpath(ori_out)
            gt_out = os.path.normpath(gt_out)
            # binarize gt: whatever isn't background, is skin
            if db == 4 or db == 3: # Uchile/UW: white background
                b = 255
                g = 255
                r = 255
                lower_val = (b, g, r)
                upper_val = lower_val
                # Threshold the image to get only selected colors
                mask = cv2.inRange(gt_im, lower_val, upper_val)
                # what isn't bg is white
                sk = cv2.bitwise_not(mask)
                gt_im = sk
            else: # background = 180,180,180
                b = 180
                g = 180
                r = 180
                lower_val = (b, g, r)
                upper_val = lower_val
                # Threshold the image to get only selected colors
                mask = cv2.inRange(gt_im, lower_val, upper_val)
                # what isn't bg is white
                sk = cv2.bitwise_not(mask)
                gt_im = sk
            # save processing 
            cv2.imwrite(ori_out, ori_im)
            cv2.imwrite(gt_out, gt_im)

            skintone = entry['skintone']

            note = 'nd'
            for entry in file3c:
                ori_csv = entry.split(csv_sep)[0]
                ori_csv = os.path.normpath(ori_csv)
                note_csv = entry.split(csv_sep)[2]
                
                if ori_out == ori_csv: # matched
                    note = note_csv
                    break
            
            out.write(f"{ori_out}{csv_sep}{gt_out}{csv_sep}{note}{csv_sep}{skintone}\n")

def get_training_and_testing_sets(file_list: list, split: float = 0.7):
    print(file_list)
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing

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