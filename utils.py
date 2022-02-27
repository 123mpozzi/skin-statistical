import os
from shutil import copyfile
from random import shuffle
from pathlib import Path
import xxhash

# CSV separator used into data.csv file
csv_sep = '?'

# Return a list of paths in the format: (ori_image_filename.ext, gt_image_filename.ext)
# matching the given strings.
# Paths are read from a 3-column CSV file.
def match_paths(csv_file: str, matches: list, match_col: int) -> list:
    file = open(csv_file)
    file3c = file.read().splitlines() # 3-column file
    file.close()

    files = []

    for entry in file3c:
        ori_path = entry.split(csv_sep)[0]
        gt_path = entry.split(csv_sep)[1]
        match_string = entry.split(csv_sep)[match_col]
        
        if len(matches) > 0: # there is a filter
            if match_string in matches:
                files.append((ori_path, gt_path))
        else: # if notes is empty, return all paths
            files.append((ori_path, gt_path))
    
    list_str = '<all>'
    if len(matches) > 0:
        list_str = ', '.join([str(elem) for elem in matches])

    if len(files) == 0:
        exit(f'No paths found matching: {list_str}')
    else:
        print(f'Found {len(files)} paths matching: {list_str}')
    
    return files

# Filter CSV dataset file to get only the TRAINING+VALIDATION split lines
def get_train_paths(csv_file) -> list:
    return match_paths(csv_file, ('tr', 'va'), 2)

# Filter CSV dataset file to get only the VALIDATION split lines
def get_val_paths(csv_file) -> list:
    return match_paths(csv_file, ('va'), 2)

# Filter CSV dataset file to get only the TESTING split lines
def get_test_paths(csv_file) -> list:
    return match_paths(csv_file, ('te'), 2)

def get_all_paths(csv_file) -> list:
    return match_paths(csv_file, (), 2)

def count_notes(csv_file: str, mode: str):
    if mode == 'train':
        matches = ('tr', 'va')
    else:
        matches = ('te')

    return len(match_paths(csv_file, matches, 2))

def count_skintones(csv_file: str, skintone: str):
    return len(match_paths(csv_file, (skintone), 3))


# Update the notes of lines with matching skintone in the Schmugge CSV file
# In training mode, lines are shuffled and follow given (VA,TE,TR) splits
# In testing mode, lines are all assigned to the TE split
# skintone may be: 'dark', 'light', 'medium'
def update_schmugge(csv_file: str, skintone: str, mode = 'train', val_percent = .15, test_percent = .15):
    # read the images CSV
    file = open(csv_file)
    file4c = file.read().splitlines()
    file.close()

    if mode == 'train':
        # randomize
        shuffle(file4c)
        # calculate splits length
        total_items = count_skintones(csv_file, skintone) # total items to train/val/test on
        va_amount = round(total_items * val_percent)
        te_amount = round(total_items * test_percent)
        # counters
        jva = 0
        jte = 0

    # rewrite csv file
    with open(csv_file, 'w') as out:
        for entry in file4c:
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]
            skint = entry.split(csv_sep)[3]

            if skint != skintone: # should not be filtered
                note = 'nd'
                out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{csv_sep}{skint}\n")
            else: # should be in the filter
                if mode == 'train': # if it is a training filter
                    if jva < va_amount: # there are still places left to be in validation set
                        note = 'va'
                        jva += 1
                    elif jte < te_amount: # there are still places left to be in test set
                        note = 'te'
                        jte += 1
                    else: # no more validation places to sit in, go in train set
                        note = 'tr'
                else: # if it is a testing filter, just place them all in test set
                    note = 'te'
                
                out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{csv_sep}{skintone}\n")

# load a schmugge skintone split by replacing the data.csv file
def load_skintone_split(skintone):
    csv_file = './dataset/Schmugge/data.csv'
    os.remove(csv_file)

    if skintone == 'light':
        print(f'Loading skintone split: {skintone}')
        copyfile('./dataset/Schmugge/light2305_1420.csv', csv_file) # TODO: now there is just light.csv
    elif skintone == 'medium':
        print(f'Loading skintone split: {skintone}')
        copyfile('./dataset/Schmugge/medium2305_1323.csv', csv_file)
    elif skintone == 'dark':
        print(f'Loading skintone split: {skintone}')
        copyfile('./dataset/Schmugge/dark2305_1309.csv', csv_file)
    else:
        print(f'Invalid skintone type: {skintone}')


# Modify the CSV file to allow benchmarking using the starting images of ECU dataset
# Benchmark images will be assigned to the testing set
def prepare_benchmark_set(csv_file, amount = 15):
    # read the images CSV
    file = open(csv_file)
    file3c = file.read().splitlines()
    file.close()

    # get the benchmark images
    filenames = []
    for i in range(amount):
        istr = str(i).zfill(2)
        filenames.append(f'im000{istr}') # ECU filenames format

    # rewrite csv file, keep only benchmark images as testing set
    with open(csv_file, 'w') as out:
        for entry in file3c:
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]
            note = 'tr'

            ori_basename = os.path.basename(ori_path)
            ori_filename = os.path.splitext(ori_basename)[0]

            if ori_filename in filenames:
                note = 'te'
                
            out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}\n")

def xxhash_checksum(filePath):
    m = xxhash.xxh3_64()

    with open(filePath, 'rb') as fh:
        #m = hashlib.md5()
        while True:
            data = fh.read(8192)
            if not data:
                break
            m.update(data)
        return m.hexdigest()

# Credit to https://stackoverflow.com/a/54477583
def hash_update_from_dir(directory, hash):
    assert Path(directory).is_dir()
    for path in sorted(Path(directory).iterdir(), key=lambda p: str(p).lower()):
        hash.update(path.name.encode())
        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash.update(chunk)
        elif path.is_dir(): # is recursive
            hash = hash_update_from_dir(path, hash)
    return hash

def hash_dir(directory):
    hash = xxhash.xxh3_64()
    return hash_update_from_dir(directory, hash).hexdigest()
