from utils.skin_dataset import skin_dataset, SingletonMeta
import os
from random import shuffle
from shutil import copyfile
import cv2
from math import floor

class Schmugge(skin_dataset, metaclass=SingletonMeta): # TODO: extend again with light,dark(),medium() ?
    def __init__(self):
        super().__init__('Schmugge')
        # paths
        self.data_manager = os.path.join(self.dir, 'data', '.config.SkinImManager')
        self.import_csv = os.path.join(self.dir, 'schmugge_import.csv')
        #self.images = os.path.join(self.dir, 'data', 'data')
        self.gt = os.path.join(self.dir, 'data', 'data')
        self.ori = os.path.join(self.dir, 'data', 'data')
        self.new_gt = os.path.join(self.dir, 'newdata', 'gt')
        self.new_ori = os.path.join(self.dir, 'newdata', 'ori')

        # images with gt errors, aa69 is also duplicated in the config file
        self.blacklist = ('aa50.gt.d3.pgm', 'aa69.gt.d3.pgm', 'dd71.gt.d3.pgm', 'hh54.gt.d3.pgm')

        # Skintone values
        self.sk_light = 'light'
        self.sk_medium = 'medium'
        self.sk_dark = 'dark'
        self.skintones = (self.sk_dark, self.sk_medium, self.sk_light)
    
    def assert_skintone(self, skintone: str):
        assert skintone in self.skintones, f'Invalid skintone: {skintone}'

    def count_skintones(self, skintone: str) -> int:
        self.assert_skintone(skintone)
        return len(self.match_paths(self.csv, (skintone), 3))
    
    # Update the notes of lines with matching skintone in the Schmugge CSV file
    # In training mode, lines are shuffled and follow given (VA,TE,TR) splits
    # In testing mode, lines are all assigned to the TE split
    # skintone may be: 'dark', 'light', 'medium'
    def update_notes(self, skintone: str, train_mode: bool = True, val_percent = .15, test_percent = .15):
        self.assert_skintone(skintone)
        # read the images CSV
        file_content = self.read_csv()

        if train_mode:
            # randomize
            shuffle(file_content)
            # calculate splits length
            total_items = self.count_skintones(skintone) # total items to train/val/test on
            va_amount = round(total_items * val_percent)
            te_amount = round(total_items * test_percent)
            # counters
            jva = 0
            jte = 0

        # rewrite csv file
        with open(self.csv, 'w') as out:
            for entry in file_content:
                csv_fields = self.split_csv_fields(entry)
                ori_path = csv_fields[0]
                gt_path = csv_fields[1]
                skint = csv_fields[3]

                if skint != skintone: # should not be filtered
                    note = self.not_defined
                    out.write(self.to_csv_row(ori_path, gt_path, note, skint))
                else: # should be in the filter
                    if train_mode: # if it is a training filter
                        if jva < va_amount: # there are still places left to be in validation set
                            note = self.nt_validation
                            jva += 1
                        elif jte < te_amount: # there are still places left to be in test set
                            note = self.nt_testing
                            jte += 1
                        else: # no more validation places to sit in, go in train set
                            note = self.nt_training
                    else: # if it is a testing filter, just place them all in test set
                        note = self.nt_testing
                    
                    out.write(self.to_csv_row(ori_path, gt_path, note, skintone))
    
    # Load a skintone split by replacing the data.csv file
    def load_skintone_split(self, skintone):
        self.assert_skintone(skintone)
        os.remove(self.csv)

        if skintone == self.sk_light:
            print(f'Loading skintone split: {skintone}')
            copyfile('./dataset/Schmugge/light2305_1420.csv', self.csv) # TODO: now there is just 'light.csv' not 2305
        elif skintone == self.sk_medium:
            print(f'Loading skintone split: {skintone}')
            copyfile('./dataset/Schmugge/medium2305_1323.csv', self.csv)
        elif skintone == self.sk_dark:
            print(f'Loading skintone split: {skintone}')
            copyfile('./dataset/Schmugge/dark2305_1309.csv', self.csv)
        else:
            print(f'Invalid skintone type: {skintone}')
    
    def get_training_and_testing_sets(self, file_list: list, split: float = 0.7):
        print(file_list)
        split_index = floor(len(file_list) * split)
        training = file_list[:split_index]
        testing = file_list[split_index:]
        return training, testing
    
    # from schmugge custom config (.config.SkinImManager) to a list of dict structure
    def read_data_manager(self) -> list: # also prepare the csv
        data = []
        
        with open(self.data_manager) as f:
            start = 0
            i = 0
            tmp = {}
            for line in f:
                blacklisted = False

                # skip first 2 lines
                if start < 2:
                    start += 1
                    continue
                
                if line: # line not empty
                    line = line.rstrip() # remove End Of Line (\n)

                    if i == 2: # skin tone type
                        skin_type = int(line)
                        if skin_type == 0:
                            tmp['skintone'] = self.sk_light
                        elif skin_type == 1:
                            tmp['skintone'] = self.sk_medium
                        elif skin_type == 2:
                            tmp['skintone'] = self.sk_dark
                        else:
                            tmp['skintone'] = self.not_defined
                    elif i == 3: # db type
                        tmp['db'] = line
                    elif i == 8: # ori
                        tmp['ori'] = os.path.join(self.ori, line)
                    elif i == 9: # gt
                        tmp['gt'] = os.path.join(self.gt, line)
                        if line in self.blacklist:
                            blacklisted = True
                    
                    # update image counter
                    i += 1
                    if i == 10: # 10 lines read, prepare for next image data
                        if not blacklisted:
                            data.append(tmp)
                        tmp = {}
                        i = 0
        
        print(f'Schmugge custom config read correctly, found {len(data)} images')
        return data
    
    # From schmugge list of dicts structure (and eventually csv_file with pre-defined splits)
    # to csv file and processed images
    def gen_csv(self, predefined: bool = True):
        manager_data = self.read_data_manager()
        # prepare new ori and gt dirs
        os.makedirs(self.new_ori, exist_ok=True)
        os.makedirs(self.new_gt, exist_ok=True)

        with open(self.csv, 'w') as out:
            if not predefined:
                shuffle(manager_data) # randomize
                # 70% train, 15% val, 15% test
                train_files, test_files = self.get_training_and_testing_sets(manager_data)
                test_files, val_files = self.get_training_and_testing_sets(test_files, split=.5)


            for entry in manager_data:
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
                ori_out = os.path.join(self.new_ori, ori_filename + '.png')
                gt_out = os.path.join(self.new_gt, gt_filename + '.png')
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

                if predefined:
                    note = self.not_defined
                    for entry in self.read_csv(self.import_csv):
                        csv_fields = self.split_csv_fields(entry)
                        ori_csv = csv_fields[0]
                        ori_csv = os.path.normpath(ori_csv)
                        note_csv = csv_fields[2]
                        
                        if ori_out == ori_csv: # matched
                            note = note_csv
                            break
                else:
                    note = self.nt_testing
                    if entry in train_files:
                        note = self.nt_training
                    elif entry in val_files:
                        note = self.nt_validation
                
                out.write(self.to_csv_row(ori_out, gt_out, note, skintone))