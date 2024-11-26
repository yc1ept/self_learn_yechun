from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

writer =SummaryWriter("logs")

img_path= "../练手数据集/train/ants_image/0013035.jpg"
#img_path = "C:\\Users\\admin\\Desktop\\图片\\QQ截图20231231224843.png"
img = Image.open(img_path).convert("RGB")
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(tensor_img)
print(img.size)
trans_resize=transforms.Resize([100,200])
img_resize=trans_resize(tensor_img)#Resize并不要求tensor格式，可以先resize再totensor
#img_resize2=trans_resize(tensor_img)
trans_resize2=transforms.Resize(100)
trans_compose = transforms.Compose([trans_resize2,tensor_trans])
img_resize2=trans_compose(img)

trans_random=transforms.RandomCrop(100)
img_random=trans_random(img)
img_random=tensor_trans(img_random)

writer.add_image("image_norm",img_norm)
writer.add_image("image_resize",img_resize)
writer.add_image("image_resize2",img_resize2)
writer.add_image("image_random",img_random)
#writer.add_image("ant",img_norm)
writer.close()