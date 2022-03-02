import os
from utils.skin_dataset import skin_dataset, SingletonMeta

class HGR(skin_dataset, metaclass=SingletonMeta):
    '''
    Hand Gesture Recognition (2014) organizes 1558 hand gesture
    images in three subparts, two of which include very
    high-resolution images, but also downscaled alternatives.
    `http://sun.aei.polsl.pl/~mkawulok/gestures/`
    ---
    Kawulok, M., Kawulok, J., Nalepa, J., & Smolka, B. (2014). Self-adaptive algorithm for
    segmenting skin regions. EURASIP Journal on Advances in Signal Processing, 2014(1).
    https://doi.org/10.1186/1687-6180-2014-170
    '''
    def __init__(self):
        # HGR_small because it uses downscaled versions of HGR2A and HGR2B
        # provided by the dataset author on their website
        super().__init__('HGR_small')
        self.import_csv = os.path.join(self.dir, 'HGR_data.csv')
    
    def get_parts(self) -> list:
        return (HGR1(), HGR2A(), HGR2B())
    
    def reset(self, predefined: bool = True) -> str:
        trace = ''

        # HGR is composed of three parts
        try:
            os.remove(self.csv)
        except OSError:
            pass

        for db in self.get_parts():
            newtrace = db.reset(predefined=predefined, append=True)
            if newtrace: # if string is not empty
                trace = trace + newtrace + '\n'
        
        if not predefined:
            self.randomize() # HGR does not have defined splits itself
        
        return trace
    
class HGR1(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__(HGR().name)
        self.name = 'HGR1'
        self.gt = os.path.join(self.dir, 'hgr1_skin', 'skin_masks')
        self.ori = os.path.join(self.dir, 'hgr1_images', 'original_images')
        self.new_gt = os.path.join(self.dir, 'hgr1_newskin')
        self.gt_process = 'bg=255_255_255,png'
        self.import_csv = os.path.join(self.dir, 'HGR_data.csv')

class HGR2A(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__(HGR().name)
        self.name = 'HGR2A'
        self.gt = os.path.join(self.dir, 'hgr2a_skin', 'skin_masks')
        self.ori = os.path.join(self.dir, 'hgr2a_images', 'original_images')
        self.new_gt = os.path.join(self.dir, 'hgr2a_newskin')
        self.gt_process = 'bg=255_255_255,png'
        self.import_csv = os.path.join(self.dir, 'HGR_data.csv')

class HGR2B(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__(HGR().name)
        self.name = 'HGR2B'
        self.gt = os.path.join(self.dir, 'hgr2b_skin', 'skin_masks')
        self.ori = os.path.join(self.dir, 'hgr2b_images', 'original_images')
        self.new_gt = os.path.join(self.dir, 'hgr2b_newskin')
        self.gt_process = 'bg=255_255_255,png'
        self.import_csv = os.path.join(self.dir, 'HGR_data.csv')
