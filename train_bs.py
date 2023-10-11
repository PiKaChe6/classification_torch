from data_generator_black_skin import BSDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import time
import torch
from models.resnet import resnet50
from models.resnext import resnext50
from models.convnext import convnext_small, convnext_tiny

if __name__ == '__main__':
    main_path = 'dataset/unwrap_plus'

    save_weight_path = 'weights/convnext50_softmax.pth'
    crop_size = (224, 224)
    divider = 6
    data_augmentation_times = 3

    train_dataset = BSDataset(main_path, mode='train', crop_size=crop_size, divider=divider,
                                   data_augmentation_times=data_augmentation_times)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

    test_dataset = BSDataset(main_path, mode='test', crop_size=crop_size, divider=divider,
                              data_augmentation_times=0)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)

    model_name = 'resnet'
    feature_extract = True


    ## 通过这样的操作，我们可以利用预训练的ResNet-50模型的特征提取能力，并根据自己的需求定制全连接层来适应特定的任务，例如分类任务中的102个类别。
    # model_ft = resnet50()
    # model_ft = resnext50()
    model_ft = convnext_tiny(pretrained=False,in_22k=False)

    # ## 使用model_ft.fc.in_features获取了ResNet-50模型最后一层的输入大小（即特征的维度）并将其赋值给num_ftrs变量。
    # num_ftrs = model_ft.fc.in_features  # 获取输出层的输入
    # ## 创建了一个新的全连接层，使用nn.Linear(num_ftrs, 102)，其中num_ftrs是输入大小，102是输出大小。
    # ## 将新的全连接层替换掉ResNet-50模型中原来的全连接层
    # model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102))

    # input_size = 224  # 输入图片的大小

    # 优化器设置
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.5)  # 学习率每7个epoch衰减成原来的1/10
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
    for epoch in range(100):
        print("epoch %d " % epoch)
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            ## inputs和labels通过data[0]和data[1]获取，然后使用.to(device)将它们移动到指定的设备（GPU或CPU）上进行计算。
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer_ft.zero_grad() # 先把所有的导数置为0

            # 把input放入网络中
            outputs = model_ft(inputs)
            # 这段代码是计算模型输出outputs和真实标签labels之间的损失值。criterion是定义的损失函数，
            # 通常用于衡量模型输出与真实标签之间的差异。在这个示例中，使用的是交叉熵损失函数nn.CrossEntropyLoss()。
            # print(outputs)
            # print(labels.long())
            loss = criterion(outputs, labels.long())
            loss.backward()
            # 确定跟新梯度的方法
            optimizer_ft.step()
            # running_loss += loss.item()是用于累加每个训练批次的损失值。
            # 在训练过程中，我们通常会计算每个批次的损失，并将其累加到running_loss变量中，以便在训练结束时计算平均损失。
            # loss.item()用于获取损失值的标量表示。在PyTorch中，损失值通常是一个包含一个元素的张量，
            # 而item()方法用于提取该张量中的标量值。然后，将这个标量值加到running_loss上，以便进行累加。
            running_loss += loss.item()
            # 表示已经累积了200个批次的损失值，此时可以计算平均损失值并打印出来。
            if i % 50 == 49:
                print('[%d, %5d] loss: %.4f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
        #### 计算validation loss
        #### 计算validation accuracy
        val_loss = 0.0
        num_val_batches = 1
        correct = 0
        total = 0

        # 关闭梯度计算
        with torch.no_grad():
            # 迭代验证集的批次
            for val_data in test_loader:
                inputs, labels = val_data[0].to(device), val_data[1].to(device)

                # 前向传播计算预测值
                outputs = model_ft(inputs)

                # 计算损失值
                loss = criterion(outputs, labels.long())
                # 计算预测标签
                # outputs.data在第1维度（列维度）上的最大值的函数。
                # 这个函数返回两个张量，第一个张量是最大值，第二个张量是最大值所在的索引。
                # 0是黑皮，1是无黑皮
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                # 累积验证损失值
                val_loss += loss.item()
                num_val_batches += 1

        # 计算平均验证损失值
        acc = 100 * correct / total
        avg_val_loss = val_loss / num_val_batches
        print("accuracy% .3f" % acc)
        print("average loss %.5f " % avg_val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("saving weights with val_loss %.5f " % avg_val_loss)
            torch.save(model_ft.state_dict(), save_weight_path)

    print('Finished training')
    # 保存模型权重






