import albumentations as aug
import cv2
import os, sys
from PIL import Image
from tqdm import tqdm
from math import floor
import numpy as np

csv_sep = '?'


def is_similar(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())

# apply Data Augmentation on all the TRAINING split defined by a CSV FILE (the file will be then updated)
def aug_schmugge(csv_file: str):
    for i in range(3): # hflip, rotate, shiftscalerotate

        # load the csv file to update
        file = open(csv_file)
        file4c = file.read().splitlines()
        file.close()

        with open(csv_file, 'w') as out:
            for entry in tqdm(file4c):
                ori_path = entry.split(csv_sep)[0]
                gt_path = entry.split(csv_sep)[1]
                note = entry.split(csv_sep)[2]
                skintone = entry.split(csv_sep)[3]

                ori_path = ori_path.replace(os.sep, '/')
                gt_path = gt_path.replace(os.sep, '/')

                # data aug applicato solo su TRAINING SET
                if note != 'tr':
                    out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{csv_sep}{skintone}\n")
                    continue

                ori_basename = os.path.basename(ori_path)
                gt_basename = os.path.basename(gt_path)
                ori_filename, ori_e = os.path.splitext(ori_basename)
                gt_filename, gt_e = os.path.splitext(gt_basename)

                # process images
                # load images
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
                    #aug.RandomCrop(width=256, height=256),
                    aug.HorizontalFlip(p=1),
                    #aug.Rotate(limit = 25, p=1),
                ])
                transform2 = aug.Compose([
                    #aug.RandomCrop(width=256, height=256),
                    #aug.HorizontalFlip(p=1),
                    aug.Rotate(limit = 15, p=.8),
                    #aug.RandomCrop(height, width, p=1),
                ])
                transform3 = aug.Compose([
                    #aug.RandomCrop(width=256, height=256),
                    #aug.HorizontalFlip(p=1),
                    #aug.ShiftScaleRotate(p=1),
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
                #gt_transformed = transform(image=gt_im)
                #gt_transformed_image = gt_transformed["image"]


                if is_similar(ori_im, ori_transformed_image) == False:
                    # save processing
                    cv2.imwrite(ori_out, cv2.cvtColor(ori_transformed_image, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(gt_out, cv2.cvtColor(gt_transformed_image, cv2.COLOR_RGB2BGR))
                    
                    out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{csv_sep}{skintone}\n") # rewrite normal images
                    out.write(f"{ori_out}{csv_sep}{gt_out}{csv_sep}{note}{csv_sep}{skintone}\n") # additional images


if __name__ == "__main__":
    # total arguments
    n = len(sys.argv)

    if n == 2:
        root_dir = sys.argv[1] # Schmugge root folder (where the CSV file is placed)
    else:
        exit('Usage: python aug.py <schmugge-root-folder>')
    
    csv_file = os.path.join(root_dir, 'data.csv')
    aug_schmugge(csv_file)
