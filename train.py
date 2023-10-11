from data_generator import FlowerDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import time
import torch
from models.resnet import resnet50

def check_dataset(data_loader):
    image, label = iter(data_loader).__next__()
    sample = image[0].squeeze()
    sample = sample.permute((1,2,0)).numpy()
    sample *= [0.229, 0.224, 0.225]
    sample += [0.485, 0.456, 0.406]
    plt.imshow(sample)
    plt.show()
    print(label[0].numpy())

if __name__ == '__main__':
    root_dir_train = 'dataset/train_filelist'
    ann_file_train = 'dataset/train.txt'

    root_dir_val = 'dataset/val_filelist'
    ann_file_val = 'dataset/val.txt'

    save_weight_path = 'weights/model_weights.pth'

    data_transforms = {
        'train':
            transforms.Compose([
                transforms.Resize(64),
                transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
                transforms.CenterCrop(64),  # 从中心开始裁剪
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
                transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
                transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
                # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
                transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
            ]),
        'valid':
            transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
    }

    data_train = FlowerDataset(root_dir=root_dir_train, ann_file=ann_file_train, transform=data_transforms['train'])
    data_loader_train = DataLoader(data_train, batch_size=2, shuffle=True, num_workers=0)

    data_val = FlowerDataset(root_dir=root_dir_val, ann_file=ann_file_val, transform=data_transforms['valid'])
    data_loader_val = DataLoader(data_val, batch_size=1, shuffle=True, num_workers=0)



    model_name = 'resnet'
    feature_extract = True


    ## 通过这样的操作，我们可以利用预训练的ResNet-50模型的特征提取能力，并根据自己的需求定制全连接层来适应特定的任务，例如分类任务中的102个类别。
    model_ft = models.resnet50()

    ## 使用model_ft.fc.in_features获取了ResNet-50模型最后一层的输入大小（即特征的维度）并将其赋值给num_ftrs变量。
    num_ftrs = model_ft.fc.in_features  # 获取输出层的输入
    ## 创建了一个新的全连接层，使用nn.Linear(num_ftrs, 102)，其中num_ftrs是输入大小，102是输出大小。
    ## 将新的全连接层替换掉ResNet-50模型中原来的全连接层
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102))

    input_size = 64  # 输入图片的大小

    # 优化器设置
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)  # 学习率每7个epoch衰减成原来的1/10
    criterion = nn.CrossEntropyLoss()

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []

    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ft.to(device)

    best_val_loss = float('inf')
    print("start training...")
    for epoch in range(30):
        print(" epoch  ")
        print( epoch)
        running_loss = 0.0
        for i, data in enumerate(data_loader_train, 0):
            ## inputs和labels通过data[0]和data[1]获取，然后使用.to(device)将它们移动到指定的设备（GPU或CPU）上进行计算。
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer_ft.zero_grad() # 先把所有的导数置为0

            # 把input放入网络中
            outputs = model_ft(inputs)
            # 这段代码是计算模型输出outputs和真实标签labels之间的损失值。criterion是定义的损失函数，
            # 通常用于衡量模型输出与真实标签之间的差异。在这个示例中，使用的是交叉熵损失函数nn.CrossEntropyLoss()。
            loss = criterion(outputs, labels)
            loss.backward()
            # 确定跟新梯度的方法
            optimizer_ft.step()
            # running_loss += loss.item()是用于累加每个训练批次的损失值。
            # 在训练过程中，我们通常会计算每个批次的损失，并将其累加到running_loss变量中，以便在训练结束时计算平均损失。
            # loss.item()用于获取损失值的标量表示。在PyTorch中，损失值通常是一个包含一个元素的张量，
            # 而item()方法用于提取该张量中的标量值。然后，将这个标量值加到running_loss上，以便进行累加。
            running_loss += loss.item()
            # 表示已经累积了200个批次的损失值，此时可以计算平均损失值并打印出来。
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        #### 计算validation loss
        val_loss = 0.0
        num_val_batches = 1

        # 关闭梯度计算
        print("Validation...")
        with torch.no_grad():
            # 迭代验证集的批次
            for val_data in data_loader_val:
                inputs, labels = val_data[0].to(device), val_data[1].to(device)

                # 前向传播计算预测值
                outputs = model_ft(inputs)

                # 计算损失值
                loss = criterion(outputs, labels)

                # 累积验证损失值
                val_loss += loss.item()
                num_val_batches += 1

        # 计算平均验证损失值
        avg_val_loss = val_loss / num_val_batches
        print("avg_val_loss")
        print(avg_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model_ft.state_dict(), save_weight_path)
            print("saving weights... ...")

    print('Finished training')
    # 保存模型权重






