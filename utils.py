import os

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