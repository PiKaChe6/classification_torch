import os
import numpy as np
import torch
from models.resnet import resnet50
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from models.resnext import resnext50
from models.convnext import convnext_tiny

def _load_and_preprocess(path, divider, crop_size):
    img = Image.open(path)
    size = img.size
    if size[0] % divider != 0:
        size_new = (size[0] - size[0] % divider, size[1])
    else:
        size_new = size
    joint = Image.new("L", (size_new[0] // divider, size_new[1] * divider))
    for i in range(divider):
        loc = (0, i * size_new[1])
        img_tmp = img.crop((i * (size_new[0] // divider), 0, (i + 1) * (size_new[0] // divider), size_new[1]))
        joint.paste(img_tmp, loc)
    img = joint.resize(crop_size, Image.BICUBIC)
    output_img = np.array(img) / 255.0
    plt.imshow(output_img)
    plt.show()
    # output_img = torch.from_numpy(output_img).to(torch.float32)
    output_img = torchvision.transforms.ToTensor()(output_img)  # 转换为PyTorch张量
    output_img = output_img.float()
    # print(output_img.shape)

    # output_img =  output_img.unsqueeze(0)

    return output_img

# model = resnet50()
# model = resnext50()
model = convnext_tiny()

# 加载模型权重
# model.load_state_dict(torch.load('weights/resnet_softmax_100.pth'))
# model.load_state_dict(torch.load('weights/resnext50_softmax_100.pth'))
model.load_state_dict(torch.load('weights/convnext50_softmax.pth', map_location=torch.device('cpu')))

# main_path = 'dataset/unwrap_plus/test/black_skin/sample18/'
main_path = 'dataset/unwrap_plus/test/clean/sample47/'
fullpath = []
for img_name in os.listdir(main_path):
    fullname = main_path + img_name
    fullpath.append(fullname)
for img in fullpath:
    image = _load_and_preprocess(img, divider=6, crop_size=(224, 224))
    # input_tensor = torchvision.transforms.ToTensor()(image)
# input_batch = input_tensor.unsqueeze(0)的作用是将输入张量的维度从(C, H, W)扩展为(1, C, H, W)，
# 其中C表示通道数，H表示高度，W表示宽度。这是因为在PyTorch中，模型的输入通常是一个批次的数据，即多个样本组成的张量。
# 扩展维度后，input_batch成为一个包含单个样本的批次张量。
    input_batch = image.unsqueeze(0)
    print(input_batch.shape)

## 在评估模式下，模型会关闭一些特定的操作，例如Dropout和Batch Normalization的随机性。
# 这是因为在训练时，Dropout和Batch Normalization会引入随机性来增强模型的泛化能力，
# 但在推断时，我们希望得到确定性的结果。
## 通过调用model.eval()，模型会将所有这些随机性的操作设置为固定状态，以便在推断过程中获得一致的输出。
# 在进行推断之前，通常需要将模型设置为评估模式。
    model.eval()

## 在PyTorch中，torch.no_grad()是一个上下文管理器，用于禁用梯度计算。
# 在模型推断过程中，我们通常不需要计算梯度，因为我们只关心模型的输出结果。
# 通过使用torch.no_grad()上下文管理器，可以减少内存消耗并提高推断的速度。
    with torch.no_grad():
        output = model(input_batch)
        print(output)
        _, predicted_idx = torch.max(output, 1)
        predicted_label = predicted_idx.item()
        print(predicted_label)
