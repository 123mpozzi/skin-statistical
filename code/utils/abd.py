import os
from utils.skin_dataset import skin_dataset, SingletonMeta

class abd(skin_dataset, metaclass=SingletonMeta):
    '''
    abd-skin (2019) is a database composed of 1400 size-fixed abdominal pictures accurately
    selected to represent different ethnic groups and body mass indices. It has native test and
    train splits.
    `https://github.com/MRE-Lab-UMD/abd-skin-segmentation`
    ---
    Topiwala, A., Al-Zogbi, L., Fleiter, T., & Krieger, A. (2019). Adaptation and Evaluation
    of Deep Learning Techniques for Skin Segmentation on Novel Abdominal Dataset.
    2019 IEEE 19th International Conference on Bioinformatics and Bioengineering (BIBE).
    https://doi.org/10.1109/bibe.2019.00141
    '''
    def __init__(self):
        super().__init__('abd-skin')
    
    def get_parts(self) -> list:
        return (abd_test(), abd_train())
    
    def reset(self, predefined) -> str:
        trace = ''

        try:
            os.remove(self.csv)
        except OSError:
            pass

        for db in self.get_parts():
            newtrace = db.reset(append=True)
            if newtrace: # if string is not empty
                trace = trace + newtrace + '\n'
        
        return trace

class abd_test(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__(abd().name)
        self.name = 'abd_test'
        self.gt = os.path.join(self.dir, 'test', 'skin_masks')
        self.ori = os.path.join(self.dir, 'test', 'original_images')
        self.note = self.nt_testing

class abd_train(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__(abd().name)
        self.name = 'abd_train'
        self.gt = os.path.join(self.dir, 'train', 'skin_masks')
        self.ori = os.path.join(self.dir, 'train', 'original_images')
        self.note = self.nt_training
