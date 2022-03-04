import os
from utils.skin_dataset import skin_dataset, SingletonMeta

class Pratheepan(skin_dataset, metaclass=SingletonMeta):
    '''
    Pratheepan (2012) is composed of 78 pictures randomly sampled from the web, precisely
    annotated. It stores the pictures containing a single subject with simple backgrounds
    and images containing multiple subjects with complex backgrounds in different folders.
    `http://cs-chan.com/downloads_skin_dataset.html`
    ---
    Tan, W. R., Chan, C. S., Yogarajah, P., & Condell, J. (2012). A Fusion Approach for
    Efficient Human Skin Detection. IEEE Transactions on Industrial Informatics, 8(1), 138-147.
    https://doi.org/10.1109/tii.2011.2172451
    ---
    Osman, M. Z., Maarof, M. A., & Rohani, M. F. (2016). Improved Dynamic Threshold
    Method for Skin Colour Detection Using Multi-Colour Space. American Journal of
    Applied Sciences, 13(2), 135-144.
    https://doi.org/10.3844/ajassp.2016.135.144
    '''
    def __init__(self):
        super().__init__('Pratheepan')
    
    def get_parts(self) -> list:
        return (Pratheepan_face(), Pratheepan_family())
    
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
        
        if not predefined:
            self.randomize() # Pratheepan does not have defined splits itself

        return trace

class Pratheepan_face(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__(Pratheepan().name)
        self.name = 'Pratheepan_face'
        self.gt = os.path.join(self.dir, 'Ground_Truth', 'GroundT_FacePhoto')
        self.ori = os.path.join(self.dir, 'Pratheepan_Dataset', 'FacePhoto')

class Pratheepan_family(skin_dataset, metaclass=SingletonMeta):
    def __init__(self):
        super().__init__(Pratheepan().name)
        self.name = 'Pratheepan_family'
        self.gt = os.path.join(self.dir, 'Ground_Truth', 'GroundT_FamilyPhoto')
        self.ori = os.path.join(self.dir, 'Pratheepan_Dataset', 'FamilyPhoto')
