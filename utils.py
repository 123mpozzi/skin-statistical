import os
from shutil import copyfile
from random import shuffle

# CSV separator used into data.csv file
csv_sep = '?'

# Filter CSV dataset file to get only the TRAINING+VALIDATION split lines
def get_train_paths(csv_file) -> list:
    # read the images CSV (ori_image_filename.ext, gt_image_filename.ext)
    file = open(csv_file)
    file3c = file.read().splitlines()
    file.close()

    files = []

    for entry in file3c:
        ori_path = entry.split(csv_sep)[0]
        gt_path = entry.split(csv_sep)[1]
        note = entry.split(csv_sep)[2]
        
        if note == 'tr' or note == 'va':
            files.append((ori_path, gt_path))
    
    if len(files) == 0:
        exit('No files found for training!')
    else:
        print(f'Found training split of {len(files)} files')
    
    return files

# Filter CSV dataset file to get only the VALIDATION split lines
def get_val_paths(csv_file) -> list:
    # read the images CSV (ori_image_filename.ext, gt_image_filename.ext)
    file = open(csv_file)
    file3c = file.read().splitlines()
    file.close()

    files = []

    for entry in file3c:
        ori_path = entry.split(csv_sep)[0]
        gt_path = entry.split(csv_sep)[1]
        note = entry.split(csv_sep)[2]
        
        if note == 'va':
            files.append((ori_path, gt_path))
    
    if len(files) == 0:
        exit('No files found for validation!')
    else:
        print(f'Found training split of {len(files)} files')
    
    return files

# Filter CSV dataset file to get only the TESTING split lines
def get_test_paths(csv_file) -> list:
    # read the images CSV (ori_image_filename.ext, gt_image_filename.ext)
    file = open(csv_file)
    file3c = file.read().splitlines()
    file.close()

    files = []

    for entry in file3c:
        ori_path = entry.split(csv_sep)[0]
        gt_path = entry.split(csv_sep)[1]
        note = entry.split(csv_sep)[2]
        
        if note == 'te':
            files.append((ori_path, gt_path))
    
    if len(files) == 0:
        exit('No files found for testing!')
    else:
        print(f'Found testing split of {len(files)} files')
    
    return files


def get_all_paths(csv_file) -> list:
    # read the images CSV (ori_image_filename.ext, gt_image_filename.ext)
    file = open(csv_file)
    file3c = file.read().splitlines()
    file.close()

    files = []

    for entry in file3c:
        ori_path = entry.split(csv_sep)[0]
        gt_path = entry.split(csv_sep)[1]
        
        files.append((ori_path, gt_path))
    
    if len(files) == 0:
        exit('No files found!')
    else:
        print(f'Found {len(files)} files')
    
    return files


# Updates the csv file by modyfing the notes of lines of the given skintone
# csv must have 4 cols!
# skintone may be: 'dark', 'light', 'medium'
def csv_skintone_filter(csv_file: str, skintone: str, mode = 'train', val_percent = .15, test_percent = .15):
    # read the images CSV
    file = open(csv_file)
    file3c = file.read().splitlines()
    file.close()

    # randomize
    shuffle(file3c)

    # calculate splits length
    totalsk = csv_skintone_count(csv_file, skintone) # total items to train/val/test on
    totalva = round(totalsk * val_percent)
    totalte = round(totalsk * test_percent)
    #totaltr = totalsk - totalva
    jva = 0
    jte = 0
    #jtr = 0


    # rewrite csv file
    with open(csv_file, 'w') as out:
        for entry in file3c:
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]

            skint = entry.split(csv_sep)[3]

            if skint != skintone: # should not be filtered
                note = 'nd'
                
                out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{csv_sep}{skint}\n")
            else: # should be in the filter
                if mode == 'train': # if it is a training filter
                    if jva < totalva: # there are still places left to be in validation set
                        note = 'va'
                        jva += 1
                    elif jte < totalte: # there are still places left to be in test set
                        note = 'te'
                        jte += 1
                    else: # no more validation places to sit in, go in train set
                        note = 'tr'
                else: # if it is a testing filter, just place them all in test set
                    note = 'te'
                
                out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{csv_sep}{skintone}\n")

# Prints the total amount of items of the given skintone
def csv_skintone_count(csv_file: str, skintone: str):
    # read the images CSV
    file = open(csv_file)
    file3c = file.read().splitlines()
    file.close()

    j = 0
    # read csv file
    with open(csv_file, 'r') as out:
        for entry in file3c:
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]
            note = entry.split(csv_sep)[2]
            skint = entry.split(csv_sep)[3]

            if skint == skintone:
                j += 1
                print(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{csv_sep}{skint}")
    
    print(f"Found {j} items of type {skintone}")
    return j

# # leaves only the entries of the corresponding
# # csv must have 4 cols!
# # skintone may be: 'dark', 'light', 'medium'
# def csv_skintone_filter(csv_file: str, skintone: str, mode = 'train'):
#     # read the images CSV
#     file = open(csv_file)
#     file3c = file.read().splitlines()
#     file.close()

#     # rewrite csv file
#     with open(csv_file, 'w') as out:
#         for entry in file3c:
#             ori_path = entry.split(csv_sep)[0]
#             gt_path = entry.split(csv_sep)[1]

#             skint = entry.split(csv_sep)[3]

#             if skint != skintone: # should not be filtered
#                 if mode != 'train':
#                     if (random.random() < 0.8): # happens 80% of the time
#                         note = 'tr'
#                     else: # happens 20% of the time
#                         note = 'va'
#                 else:
#                     note = 'te'
#                     #note = entry.split(csv_sep)[2]
                
#                 #print(f"NOT {ori_path}{csv_sep}{gt_path}{csv_sep}{note}{csv_sep}{skint}\n")
#                 out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{csv_sep}{skint}\n")
#             else: # should be in the filter
#                 if mode == 'train':
#                     if (random.random() < 0.8): # happens 80% of the time
#                         note = 'tr'
#                     else: # happens 20% of the time
#                         note = 'va'
#                 else:
#                     note = 'te'
#                     #note = entry.split(csv_sep)[2]
                
#                 #print(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{csv_sep}{skintone}\n")
#                 out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{csv_sep}{skintone}\n")


# def csv_skintone_count(csv_file: str, skintone: str):
#     # read the images CSV
#     file = open(csv_file)
#     file3c = file.read().splitlines()
#     file.close()

#     j = 0
#     # rewrite csv file
#     with open(csv_file, 'r') as out:
#         for entry in file3c:
#             ori_path = entry.split(csv_sep)[0]
#             gt_path = entry.split(csv_sep)[1]
#             note = entry.split(csv_sep)[2]
#             skint = entry.split(csv_sep)[3]

#             if skint == skintone:
#                 j += 1
#                 print(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{csv_sep}{skint}")
    
#     print(f"Found {j} items of type {skintone}")


def csv_note_count(csv_file: str, mode: str):
    # read the images CSV
    file = open(csv_file)
    file3c = file.read().splitlines()
    file.close()

    j = 0
    # rewrite csv file
    with open(csv_file, 'r') as out:
        for entry in file3c:
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]
            nt = entry.split(csv_sep)[2]
            skint = entry.split(csv_sep)[3]

            notes = []
            if mode == 'train':
                notes.append("tr")
                notes.append("va")
            else:
                notes.append("te")

            if nt in notes:
                j += 1
                print(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{nt}{csv_sep}{skint}")
    
    print(f"Found {j} items of type {mode}")

# load a schmugge skintone split by replacing the data.csv file
def load_skintone_split(skintone):
    os.remove('./dataset/Schmugge/data.csv')

    if skintone == 'light':
        print(f'loading skintone split: {skintone}')
        copyfile('./dataset/Schmugge/light2305_1420.csv', './dataset/Schmugge/data.csv')
    elif skintone == 'medium':
        print(f'loading skintone split: {skintone}')
        copyfile('./dataset/Schmugge/medium2305_1323.csv', './dataset/Schmugge/data.csv')
    elif skintone == 'dark':
        print(f'loading skintone split: {skintone}')
        copyfile('./dataset/Schmugge/dark2305_1309.csv', './dataset/Schmugge/data.csv')
    else:
        print(f'skintone type invalid: {skintone}')