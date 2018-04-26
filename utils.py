from common import *

def nifti_import(path):
    img_data_nii = nib.load(path)
    img_data_nii = np.array(img_data_nii.get_data())
    return np.expand_dims(img_data_nii, axis=0)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
#        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_img(x):
    x = x.view(x.size(0), 1, 80, 80)
    return x
