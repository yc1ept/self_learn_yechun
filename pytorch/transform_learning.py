from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
#transforms.ToTensor
img_path = "../练手数据集/train/ants_image/0013035.jpg"
img=Image.open(img_path)

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor() # 创建一个totensor类实例，由于定义了call方法，所以可以将这个实例当成函数使用。
tensor_img = tensor_trans(img)

writer.add_image("Tensor image",tensor_img)
writer.close()