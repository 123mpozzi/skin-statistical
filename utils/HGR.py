from utils.skin_dataset import skin_dataset, SingletonMeta

class HGR(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__('HGR_small')
        #self.gt = os.path.join(self.dir, 'skin_masks')
        #self.ori = os.path.join(self.dir, 'origin_images')