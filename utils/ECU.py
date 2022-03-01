import os
from utils.skin_dataset import skin_dataset, SingletonMeta

class ECU(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__('ECU')
        #self.gt = os.path.join(self.dir, 'skin_masks')
        #self.ori = os.path.join(self.dir, 'origin_images')
    
    # Modify the CSV file to allow benchmarking using the starting images of ECU dataset
    # Benchmark images will be assigned to the testing set
    def prepare_benchmark_set(self, amount = 15):
        file_content = self.read_csv()

        # get the benchmark images
        filenames = []
        for i in range(amount):
            istr = str(i).zfill(2)
            filenames.append(f'im000{istr}') # ECU filenames format # TODO: not sure, this is from abd-git #TODO: what if amount has 3 cifre? - better zerofill!!!

        # rewrite csv file, keep only benchmark images as testing set
        with open(self.csv, 'w') as out:
            for entry in file_content:
                csv_fields = self.split_csv_fields(entry)
                ori_path = csv_fields[0]
                gt_path = csv_fields[1]
                note = self.nt_training

                ori_basename = os.path.basename(ori_path)
                ori_filename = os.path.splitext(ori_basename)[0]

                if ori_filename in filenames:
                    note = self.nt_testing
                    
                out.write(self.to_csv_row(ori_path, gt_path, note))