import os
from utils.skin_dataset import skin_dataset, SingletonMeta

class ECU(skin_dataset, metaclass=SingletonMeta):
    '''
    ECU (2005) is a dataset created at the Edith Cowan University consisting of 3998 pictures.
    Most of its content is half-body shots. It is also referred to as Face and Skin
    Detection Database (FSD).
    `https://documents.uow.edu.au/~phung/download.html`
    ---
    Phung, S., Bouzerdoum, A., & Chai, D. (2005). Skin segmentation using color pixel
    classification: analysis and comparison. IEEE Transactions on Pattern Analysis
    and Machine Intelligence, 27(1), 148-154.
    https://doi.org/10.1109/tpami.2005.17
    '''
    def __init__(self):
        super().__init__('ECU')
        self.gt = os.path.join(self.dir, 'skin_masks')
        self.ori = os.path.join(self.dir, 'origin_images')
        # TODO: re-generate split from Skinny ipynb and set self.import_csv = ..
    
    def reset(self, predefined: bool = True) -> str:
        trace = super().reset(predefined=predefined)
        
        if not predefined:
            self.randomize() # ECU does not have defined splits itself
        
        return trace

class ECU_bench(skin_dataset, metaclass=SingletonMeta):
    '''ECU dataset being setup for inference time evaluation'''
    def __init__(self):
        super().__init__(ECU().name)
        self.name = 'ECU_bench'
        self.csv = os.path.join(self.dir, 'ECU_bench.csv')
    
    # Create benchmark CSV file using the starting images of ECU dataset
    # Benchmark images will be assigned to the testing set
    def reset(self, amount = 15):
        file_content = self.read_csv(ECU().csv)

        # get the benchmark images
        filenames = []
        for i in range(amount):
            # Filename format may differ in other versions
            istr = str(i).zfill(5)
            filenames.append(f'im{istr}')
        
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