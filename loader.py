from common import *
from utils import *

class ADNIDataloader(data.Dataset):
    def __init__(self, data_file, data_path, transform=None):
        super(ADNIDataloader, self).__init__()

        self.file_list = []
        img_name_str = None
        
        with open(data_file,'r') as f:
            while True:
                img_name_str = f.readline()
                if img_name_str != '':
                    self.file_list.append( img_name_str[:-1] )
                else:
                    break
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, i):

        img_path = os.path.join(self.data_path, self.file_list[i])
        img = nifti_import(img_path)

        if self.transform:
            img = self.transform(img)

        # high = 1295
        # low = 845

        # img[np.nonzero(img>high)] = high
        # img[np.nonzero(img<low)] = low
        # img = 2*(img - low)/(high-low) - 1

        return img