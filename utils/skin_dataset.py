import os

class skin_dataset(object): # TODO: reset() method to re-generate data.csv
    def __init__(self, name):
        self.name = name
        # CSV separator used into data.csv file
        self.csv_sep = '?'
        # Training splits Notes
        self.nt_training = 'tr'
        self.nt_validation = 'va'
        self.nt_testing = 'te'
        self.not_defined = 'nd'
        # Paths
        self.dir = os.path.join('.', 'dataset', self.name)
        self.csv = os.path.join(self.dir, 'data.csv')
    
    def read_csv(self, csv_file = None) -> list:
        if csv_file is None:
            csv_file = self.csv
        file = open(csv_file)
        file_content = file.read().splitlines() # multi-column file
        file.close()
        return file_content
    
    def split_csv_fields(self, row: str) -> list:
        return row.split(self.csv_sep)
    
    def to_csv_row(self, *args) -> str:
        row = args[0]
        for item in args[1:]: # TODO: is 1 included or excluded?
            row += self.csv_sep
            row += item
        row += '\n'
        return row

    # Return a list of paths in the format: (ori_image_filename.ext, gt_image_filename.ext)
    # matching the given strings.
    # Paths are read from a 3-column CSV file.
    def match_paths(self, matches: list, match_col: int) -> list:
        files = []

        for entry in self.read_csv():
            ori_path = entry.split(self.csv_sep)[0]
            gt_path = entry.split(self.csv_sep)[1]
            match_string = entry.split(self.csv_sep)[match_col]
            
            if len(matches) > 0: # there is a filter
                if match_string in matches:
                    files.append((ori_path, gt_path))
            else: # if notes is empty, return all paths
                files.append((ori_path, gt_path))
        
        list_str = '<all>'
        if len(matches) > 0:
            list_str = ', '.join([str(elem) for elem in matches]) # TODO: bug: t, e instead of te, 

        if len(files) == 0:
            exit(f'No paths found matching: {list_str}')
        else:
            print(f'Found {len(files)} paths matching: {list_str}')
        
        return files

    # Filter CSV dataset file to get only the TRAINING+VALIDATION split lines
    def get_train_paths(self) -> list:
        return self.match_paths((self.nt_training, self.nt_validation), 2)

    # Filter CSV dataset file to get only the VALIDATION split lines
    def get_val_paths(self) -> list:
        return self.match_paths((self.nt_validation), 2)

    # Filter CSV dataset file to get only the TESTING split lines
    def get_test_paths(self) -> list:
        return self.match_paths((self.nt_testing), 2)

    def get_all_paths(self) -> list:
        return self.match_paths((), 2)

    def count_notes(self, mode: str) -> int:
        if mode == 'train':
            matches = (self.nt_training, self.nt_validation)
        else:
            matches = (self.nt_testing)
        return len(self.match_paths(self.csv, matches, 2))
    
    # TODO: reset() method which re-process the db


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

# TODO: singleton unit test
#s1 = ECU()
#s2 = ECU()
#if id(s1) == id(s2):
#    print("Singleton works, both variables contain the same instance.")
#else:
#    print("Singleton failed, variables contain different instances.")
