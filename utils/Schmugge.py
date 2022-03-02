from utils.skin_dataset import skin_dataset, SingletonMeta
import os
import traceback
from random import shuffle
import cv2
from tqdm import tqdm
import albumentations as aug
import numpy as np
from math import floor


class Schmugge(skin_dataset, metaclass=SingletonMeta):
    '''
    Schmugge (2007) is a facial dataset that includes 845 images taken from different databases.
    It provides several labeled information about each image and ternary ground truths.
    `https://www.researchgate.net/publication/257620282_skin_image_Data_set_with_ground_truth`
    ---
    Schmugge, S. J., Jayaram, S., Shin, M. C., & Tsap, L. V. (2007). Objective evaluation of
    approaches of skin detection using ROC analysis. Computer Vision and Image Understanding,
    108(1-2), 41-51.
    https://doi.org/10.1016/j.cviu.2006.10.009
    '''
    def __init__(self):
        super().__init__('Schmugge')
        # paths
        self.data_manager = os.path.join(self.dir, 'data', '.config.SkinImManager')
        # Use import_csv to retrieve the configuration used in the thesis
        self.import_csv = os.path.join(self.dir, 'schmugge_datacsv_model.csv')
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
    
    # Reimport the dataset
    # Generate data.csv by reprocessing the dataset content
    # Return stack trace if there are errors, else empty
    def reset(self, predefined: bool = True) -> str:
        # Catch eventual errors
        stacktrace = ''

        try:
            # From schmugge list of dicts structure (and eventually csv_file with pre-defined splits)
            # to csv file and processed images
            manager_data = self.read_data_manager()
            # prepare new ori and gt dirs
            os.makedirs(self.new_ori, exist_ok=True)
            os.makedirs(self.new_gt, exist_ok=True)

            if not predefined:
                shuffle(manager_data) # randomize
                # 70% train, 15% val, 15% test
                train_files, test_files = self.get_training_and_testing_sets(manager_data)
                test_files, val_files = self.get_training_and_testing_sets(test_files, split=.5)

            lines_count = len(manager_data)
            if self.name in self.skintones:
                lines_count = count_skintones(self, self.name)

            progress_bar = tqdm(total=lines_count, desc=f'{self.name} creating CSV file')
            with open(self.csv, 'w') as out:
                # Process images
                for entry in manager_data:
                    db = int(entry['db'])
                    ori_path = entry['ori']
                    gt_path = entry['gt']
                    skintone = entry['skintone']

                    # Resetting skintone dataset: leave only lines with corresponding skintone
                    if self.name in self.skintones and self.name != skintone:
                        continue
                    
                    ori_basename = os.path.basename(ori_path)
                    gt_basename = os.path.basename(gt_path)
                    ori_filename, _ = os.path.splitext(ori_basename)
                    gt_filename, _ = os.path.splitext(gt_basename)

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

                    # If import configuration, just process images
                    if predefined:
                        progress_bar.update()
                        continue
                    else:
                        note = self.nt_testing
                        if entry in train_files:
                            note = self.nt_training
                        elif entry in val_files:
                            note = self.nt_validation
                        
                        out.write(self.to_csv_row(ori_out, gt_out, note, skintone))
                        progress_bar.update()

            progress_bar.close()
            if predefined:
                self.import_configuration(force=True)
            print(f'(V) Dataset {self.name} import success!')
        except Exception:
            print(f'(X) Dataset {self.name} import failed!')
            stacktrace = traceback.format_exc()
        finally:
            return stacktrace
        

class dark(Schmugge, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__()
        self.name = self.sk_dark
        self.csv = os.path.join(self.dir, 'dark.csv')
        # Note: the thesis used data-augmentation for dark
        # Images are in newdata folder
        self.import_csv = os.path.join(self.dir, 'dark2305_1309.csv')
        self.import_configuration()
    
    def reset(self, predefined: bool = True) -> str:
        trace = super().reset(predefined = predefined)
        # filter_csv(..) is needed because with the Schmugge csv_file,
        # get_all_paths() would return also paths regarding other skintones
        filter_csv(self, self.name)

        # Augment if not importing configuration
        if not predefined:
            self.augment()
        return trace
    
    # Apply Data Augmentation on all the TRAINING split defined by CSV FILE (the file will be updated)
    def augment(self):
        # Apply three transformations: hflip, rotate, shiftscalerotate
        for i in range(3):
            # Load the csv file to update
            file_content = self.read_csv()

            with open(self.csv, 'w') as out:
                progress_bar = tqdm(total=len(file_content), desc=f'{self.name}   augmenting data')
                for entry in file_content:
                    csv_fields = self.split_csv_fields(entry)
                    ori_path = csv_fields[0]
                    gt_path = csv_fields[1]
                    note = csv_fields[2]
                    skintone = csv_fields[3]

                    ori_path = ori_path.replace(os.sep, '/')
                    gt_path = gt_path.replace(os.sep, '/')

                    # Data augmentation is applied only on TRAINING SET
                    if note != 'tr':
                        out.write(self.to_csv_row(ori_path, gt_path, note, skintone))
                        continue

                    ori_basename = os.path.basename(ori_path)
                    gt_basename = os.path.basename(gt_path)
                    ori_filename, _ = os.path.splitext(ori_basename)
                    gt_filename, _ = os.path.splitext(gt_basename)


                    # Process images

                    # Load images
                    ori_im = cv2.imread(ori_path)
                    ori_im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB) # BGR to RGB
                    gt_im = cv2.imread(gt_path)
                    gt_im = cv2.cvtColor(gt_im, cv2.COLOR_BGR2RGB)
                    height, width, _ = ori_im.shape
                    height = floor(height *.8)
                    width = floor(width *.8)
                    # png
                    ori_out = os.path.join(os.path.dirname(ori_path), ori_filename + f'_aug{i}.png')
                    gt_out = os.path.join(os.path.dirname(gt_path), gt_filename + f'_aug{i}.png')
                    # on windows the path slashes are mixed
                    #ori_out = os.path.normpath(ori_out)
                    #gt_out = os.path.normpath(gt_out)

                    ori_out = ori_out.replace(os.sep, '/')
                    gt_out = gt_out.replace(os.sep, '/')


                    # Augment Data

                    # Declare an augmentation pipeline
                    transform1 = aug.Compose([
                        aug.HorizontalFlip(p=1),
                    ])
                    transform2 = aug.Compose([
                        aug.Rotate(limit = 15, p=.8),
                    ])
                    transform3 = aug.Compose([
                        aug.RandomCrop(height, width, p=.8),
                    ])

                    # Augment an image
                    if i == 0:
                        transformed = transform1(image=ori_im, mask=gt_im)
                    elif i == 1:
                        transformed = transform2(image=ori_im, mask=gt_im)
                    else:
                        transformed = transform3(image=ori_im, mask=gt_im)
                    ori_transformed_image = transformed["image"]
                    gt_transformed_image = transformed["mask"]


                    if is_similar(ori_im, ori_transformed_image) == False:
                        # Save processing
                        cv2.imwrite(ori_out, cv2.cvtColor(ori_transformed_image, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(gt_out, cv2.cvtColor(gt_transformed_image, cv2.COLOR_RGB2BGR))
                        
                        out.write(self.to_csv_row(ori_path, gt_path, note, skintone)) # rewrite normal images
                        out.write(self.to_csv_row(ori_out, gt_out, note, skintone)) # additional images
                    
                    progress_bar.update()
                progress_bar.close()

class light(Schmugge, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__()
        self.name = self.sk_light
        self.csv = os.path.join(self.dir, 'light.csv')
        self.import_csv = os.path.join(self.dir, 'light2305_1420.csv')
        self.import_configuration()
    
    def reset(self, predefined: bool = True) -> str:
        trace = super().reset(predefined = predefined)
        filter_csv(self, self.name)
        return trace

class medium(Schmugge, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__()
        self.name = self.sk_medium
        self.csv = os.path.join(self.dir, 'medium.csv')
        self.import_csv = os.path.join(self.dir, 'medium2305_1323.csv')
        self.import_configuration()

    def reset(self, predefined: bool = True) -> str:
        trace = super().reset(predefined = predefined)
        filter_csv(self, self.name)
        return trace


def is_similar(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())

def count_skintones(db: skin_dataset, skintone: str) -> int:
    assert skintone in Schmugge().skintones, f'Invalid skintone: {skintone}'
    return len(db.match_paths((skintone), 3, csv_file=Schmugge().import_csv))

# Filter the Schmugge CSV file to leave only lines with given skintone
def filter_csv(db: Schmugge, skintone):
    file_content = db.read_csv()

    with open(db.csv, 'w') as out:
        for entry in file_content:
            csv_fields = db.split_csv_fields(entry)
            if csv_fields[3] == skintone:
                out.write(db.to_csv_row(*csv_fields))
