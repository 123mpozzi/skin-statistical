import imghdr
import os
import re
import traceback
from math import floor
from random import shuffle
from shutil import copyfile

import cv2
from tqdm import tqdm

from utils.logmanager import *


class skin_dataset(object):
    '''Abstraction of a skin dataset'''
    def __init__(self, name):
        self.name = name
        # CSV separator used into data.csv file
        self.csv_sep = '?'
        # Training splits Notes
        self.nt_training = 'tr'
        self.nt_validation = 'va'
        self.nt_testing = 'te'
        self.not_defined = 'nd'
        self.available_notes = (self.nt_training, self.nt_validation, self.nt_testing, self.not_defined)
        # Paths
        current_dir = os.getcwd()
        project_dir = os.path.basename(os.path.abspath(os.path.join(current_dir, '..')))
        assert project_dir == 'skin-statistical', 'wrong cwd: ' + project_dir
        self.dir = os.path.join('..', 'dataset', self.name)
        self.csv = os.path.join(self.dir, 'data.csv')

        # This class should be used as an abstraction
        # Following values are used in the dataset import phase
        self.gt = None
        self.ori = None
        self.new_gt = None # if not None means the db has been processed
        self.new_ori = None # if not None means the db has been processed
        self.gt_process = None
        self.ori_process = None
        self.gt_format = None
        self.ori_format = None
        self.note = None
        
        # Use import_csv to retrieve the configuration used in the thesis
        self.import_csv = None
    
    def read_csv(self, csv_file = None) -> list:
        '''Return the multi-column content of the dataset CSV file'''
        if csv_file is None:
            csv_file = self.csv
        
        # Try accessing CSV file
        # May fail: eg. file does not exist
        try:
            file = open(csv_file)
            file_content = file.read().splitlines() # multi-column file
            file.close()
        except Exception:
            print(traceback.format_exc())
            critical('Error on accessing ' + csv_file)
            exit()
        
        return file_content
    
    def split_csv_fields(self, row: str) -> list:
        '''Split the fields of a row taken from the dataset CSV file'''
        return row.split(self.csv_sep)
    
    def to_csv_row(self, *args) -> str:
        '''Prepare data to be insterted as a row into the dataset CSV file'''
        row = args[0]
        for item in args[1:]:
            row += self.csv_sep
            row += item
        row += '\n'
        return row

    def to_basenames(self, row: str) -> list:
        '''Return the given row with basenames instead of paths for ori and gt'''
        csv_fields = self.split_csv_fields(row)
        ori_basename = os.path.basename(csv_fields[0])
        gt_basename = os.path.basename(csv_fields[1])
        return self.to_csv_row(ori_basename, gt_basename, *csv_fields[2:])

    def match_paths(self, matches: list, match_col: int, csv_file = None) -> list:
        '''
        Return a list of paths formatted as a tuple:
        (ori_image_filename.ext, gt_image_filename.ext) matching the given strings.

        Paths are read from a 3-column CSV file.
        '''
        files = []

        if csv_file is None:
            csv_file = self.csv

        # matches variable to string
        list_str = '<all>'
        if len(matches) > 0:
            if type(matches) is list:
                list_str = ', '.join([str(elem) for elem in matches])
            else:
                list_str = matches

        # Match paths
        for entry in self.read_csv(csv_file=csv_file):
            csv_fields = self.split_csv_fields(entry)
            ori_path = csv_fields[0]
            gt_path = csv_fields[1]
            match_string = csv_fields[match_col]
            
            if len(matches) > 0: # there is a filter
                if match_string in matches:
                    files.append((ori_path, gt_path))
            else: # if notes is empty, return all paths
                files.append((ori_path, gt_path))

        if len(files) == 0:
            critical(f'{csv_file} No paths found matching: {list_str}')
            exit()
        else:
            info(f'{csv_file} Found {len(files)} paths matching: {list_str}')
        
        return files

    def get_train_paths(self) -> list:
        '''Filter CSV dataset file to get only the TRAINING+VALIDATION split lines'''
        return self.match_paths((self.nt_training, self.nt_validation), 2)

    def get_val_paths(self) -> list:
        '''Filter CSV dataset file to get only the VALIDATION split lines'''
        return self.match_paths((self.nt_validation), 2)

    def get_test_paths(self) -> list:
        '''Filter CSV dataset file to get only the TESTING split lines'''
        return self.match_paths((self.nt_testing), 2)

    def get_all_paths(self) -> list:
        return self.match_paths((), 2)

    def get_training_and_testing_sets(self, file_list: list, split: float = 0.7):
        debug(file_list)

        split_index = floor(len(file_list) * split)
        training = file_list[:split_index]
        testing = file_list[split_index:]
        return training, testing

    def import_configuration(self, force: bool = False):
        '''Check if `self.csv` file exists: if not, try to import it from `self.import_csv`'''
        if self.import_csv is None:
            return

        # Import thesis configuration if self.csv_file does not exist
        if not os.path.isfile(self.csv):
            copyfile(self.import_csv, self.csv)
        # Force re-import even if file exists
        elif force == True:
            try:
                os.remove(self.csv)
            except OSError:
                pass
            copyfile(self.import_csv, self.csv)

    # Reset should be the only function that can modify data.csv!
    def reset(self, append: bool = False, predefined: bool = True) -> str:
        '''
        Reimport the dataset

        Generate `data.csv` by reprocessing the dataset content

        Return stack trace if there are errors, else empty
        '''
        # Required variables
        msg_none = 'Cannot re-import dataset named "{}": {} is None'
        assert self.gt is not None, msg_none.format(self.name, 'gt')
        assert self.ori is not None, msg_none.format(self.name, 'ori')
        # Non-Defined as default note
        if self.note is None:
            self.note = 'nd'
        else:
            assert self.note in self.available_notes, 'Invalid note: ' + self.note
        
        if self.gt == self.ori:
            warning('''Code is unlikely to work if original images are
            in the same directory as mask images''')

        # Catch eventual errors
        stacktrace = ''

        try:
            # Check if processing is required
            if self.ori_process is not None:
                self.new_ori = process_images(self, self.ori, self.ori_process, self.new_ori, self.ori_format)
            if self.gt_process is not None:
                self.new_gt = process_images(self, self.gt, self.gt_process, self.new_gt, self.gt_format)

            # Create the csv files
            if not predefined or self.import_csv is None:
                # Analyze the dataset and create the csv files
                analyze_dataset(self, append)
            else:
                self.import_configuration(force=True)
            
            info(f'Dataset {self.name} import success!')
        except Exception:
            error(f'Dataset {self.name} import failed!')
            stacktrace = traceback.format_exc()
            print(stacktrace)
        finally:
            return stacktrace
    
    def randomize(self):
        '''Randomize training, validation, and testing splits inside CSV file'''
        file_content = self.read_csv()
        shuffle(file_content) # randomize

        # 70% train, 15% val, 15% test
        train_files, test_files = self.get_training_and_testing_sets(file_content)
        test_files, val_files = self.get_training_and_testing_sets(test_files, split=.5)

        with open(self.csv, 'w') as out:
            for entry in file_content:
                csv_fields = self.split_csv_fields(entry)
                csv_fields[0] = os.path.normpath(csv_fields[0])
                csv_fields[1] = os.path.normpath(csv_fields[1])
                
                note = self.nt_testing
                if entry in train_files:
                    note = self.nt_training
                elif entry in val_files:
                    note = self.nt_validation
                
                csv_fields[2] = note
                
                # operator * expand the list before function call
                out.write(self.to_csv_row(*csv_fields))


# Credit to https://refactoring.guru/design-patterns/singleton/python/example
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


def is_image(path: str) -> bool:
    return os.path.isfile(path) and imghdr.what(path) != None

def get_variable_filename(filename: str, format: str) -> str:
    '''Get the variable part of a given dataset filename'''
    if format is None:
        return filename

    # re.fullmatch(r'^img(.*)$', 'imgED (1)').group(1)
    # re.fullmatch(r'^(.*)-m$', 'att-massu.jpg-m').group(1)
    match =  re.fullmatch('^{}$'.format(format), filename)
    if match:
        return match.group(1)
    else:
        #print('Cannot match {} with pattern {}'.format(filename, format))
        return None

def analyze_dataset(db: skin_dataset, append: bool = False):
    '''Create CSV file containing dataset metadata (such as paths of images)'''
    # Use gt/ori if not processed, else new_gt/new_ori
    gt_dir = db.gt
    ori_dir = db.ori
    if db.new_gt is not None:
        gt_dir = db.new_gt
    if db.new_ori is not None:
        ori_dir = db.new_ori
    
    # Number of images found
    i = 0

    # Write to data file
    write_mode = 'a' if append else 'w'
    dir_content = os.listdir(gt_dir)
    progress_bar = tqdm(total=len(dir_content), desc=f'{db.name} creating CSV file')
    with open(db.csv, write_mode) as out:
        for gt_file in dir_content:
            gt_path = os.path.join(gt_dir, gt_file)
            
            # Check if current file is an image (avoid issues with files like thumbs.db)
            if is_image(gt_path):
                matched = False
                gt_name = os.path.splitext(gt_file)[0]

                gt_identifier = get_variable_filename(gt_name, db.gt_format)
                if gt_identifier is None:
                    continue
                
                for ori_file in os.listdir(ori_dir):
                    ori_path = os.path.join(ori_dir, ori_file)
                    ori_name = os.path.splitext(ori_file)[0]

                    ori_identifier = get_variable_filename(ori_name, db.ori_format)
                    if ori_identifier is None:
                        continue
                    
                    # Try to find a match (original image - gt)
                    if gt_identifier == ori_identifier:
                        out.write(db.to_csv_row(ori_path, gt_path, db.note))
                        i += 1
                        matched = True
                        break
                if not matched:
                    debug(f'No matches found for {gt_identifier}')
            else:
                debug(f'File {gt_path} is not an image')
            progress_bar.update()
    progress_bar.close()
    info(f"Found {i} images")

# Perform image-processing on a directory content
# 
# Processing Pipeline example:
#   "png,skin=255_255_255,invert"
#   skin=.. Skin-based binarization rule:
#           pixels of whatever is not skin will be set black; skin pixels will be set white
#   bg=..   Background-based binarization rule:
#           pixels of whatever is not background will be set white; background pixels will be set black
#   png     Convert the image to PNG format
# Processing operations are performed in order!
def process_images(db: skin_dataset, data_dir: str, process_pipeline: str, out_dir: str,
                im_filename_format: str) -> str:
    dir_content = os.listdir(data_dir)
    progress_bar = tqdm(total=len(dir_content), desc=f'{db.name} processing images')

    # Loop all files in the directory
    for im_basename in dir_content:
        im_path = os.path.join(data_dir, im_basename)
        im_filename = os.path.splitext(im_basename)[0]

        # Check if current file is an image (avoid issues with files like thumbs.db)
        if is_image(im_path):
            if out_dir is None:
                out_dir = os.path.join(data_dir, 'processed')

            os.makedirs(out_dir, exist_ok=True)

            im_identifier = get_variable_filename(im_filename, im_filename_format)
            if im_identifier is None:
                continue

            # Load image
            im = cv2.imread(im_path)

            # Prepare path for out image
            im_path = os.path.join(out_dir, im_basename)

            for operation in process_pipeline.split(','):
                # Binarize
                if operation.startswith('skin') or operation.startswith('bg'):
                    # Inspired from https://stackoverflow.com/a/53989391
                    bgr_data = operation.split('=')[1]
                    b,g,r = [int(i) for i in bgr_data.split('_')]
                    lower_val = (b, g, r)
                    upper_val = lower_val

                    # If 'skin': catch only skin pixels via thresholding
                    # If 'bg':   catch only background pixels via thresholding
                    mask = cv2.inRange(im, lower_val, upper_val)
                    im = mask if operation.startswith('skin') else cv2.bitwise_not(mask)
                # Invert image
                elif operation == 'invert':
                    im = cv2.bitwise_not(im)
                # Convert to png
                elif operation == 'png':
                    im_path = os.path.join(out_dir, im_filename + '.png')
                # Reload image
                elif operation == 'reload':
                    im = cv2.imread(im_path)
                else:
                    error(f'Image processing operation unknown: {operation}')

            # Save processing 
            cv2.imwrite(im_path, im)
            progress_bar.update()

    progress_bar.close()
    return out_dir
