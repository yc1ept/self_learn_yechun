from torch.utils.data import Dataset
import os
from PIL import Image
class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_path=os.listdir(self.path)
    def __getitem__(self, idx):
        img_name =self.img_path[idx]
        img_item_path =os.path.join(self.root_dir,self.label_dir,img_name)
        img =Image.open(img_item_path)
        label =self.label_dir[0:4]
        return img,label
    def __len__(self):
        return len(self.img_path)
root_dir= "../练手数据集/train"
ants_label_dir="ants_image"
bees_label_dir="bees_image"
ants=MyData(root_dir,ants_label_dir)
bees=MyData(root_dir,bees_label_dir)
train=ants+bees
ants[1]
img,label=ants[1]
print(label)
img.show()
#dir_path="练手数据集/train/ants_image"
#img_path_list=os.listdir(dir_path)#获取文件地址列表