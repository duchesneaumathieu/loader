from loader import AbstractLoader

class InstanceSegmentation(AbstractLoader):
    def __init__(self, crop=(224,224), **kwargs):
        if isinstance(crop, int): crop = (crop, crop)
        assert isinstance(crop, tuple), 'crop is not a tuple or an integer'
        assert len(crop) == 2, 'crop length must be 2'
        self.crop=crop
        
        super(InstanceSegmentation, self).__init__(**kwargs)