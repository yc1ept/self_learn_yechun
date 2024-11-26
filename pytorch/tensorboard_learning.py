from torch.utils.tensorboard import SummaryWriter
#help(SummaryWriter)
import numpy as np
from PIL import Image

writer =SummaryWriter("../logs")
img_path= "练手数据集/train/ants_image/0013035.jpg"
img =Image.open(img_path)
img_array = np.array(img)
writer.add_image("test",img_array,1,dataformats='HWC')#由于shape不对，并非要求的通道，高度，宽度为次序的格式，需要在后面加上dataformats
#writer.add_image()
#y=2x
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

writer.close()