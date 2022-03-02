import os
from utils.skin_dataset import skin_dataset, SingletonMeta

class UChile(skin_dataset, metaclass=SingletonMeta):
    '''
    UChile (2006) is a dataset of 101 images obtained from Internet and
    from digitized news videos, which was fully annotated by a human operator.
    `http://web.archive.org/web/20070707151628/http://agami.die.uchile.cl/skindiff/`
    ---
    J. Ruiz-del-Solar and R. Verschae. “SKINDIFF-Robust and fast skin segmentation”.
    Department of Electrical Engineering, Universidad de Chile, 2006.
    '''
    def __init__(self):
        super().__init__('UChile')
        self.gt = os.path.join(self.dir, 'mask')
        self.ori = os.path.join(self.dir, 'original')
        self.gt_format = '(.*)-m'
    
    def reset(self, predefined):
        super().reset(predefined=False)
        super().randomize() # UChile does not have defined splits itself
    