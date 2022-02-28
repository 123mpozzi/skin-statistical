from utils.skin_dataset import skin_dataset, SingletonMeta

class HGR(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__('HGR_small')