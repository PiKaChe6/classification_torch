import os
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
class BSDataset(Dataset):
    def __init__(self, main_path, mode, crop_size, divider, data_augmentation_times):
        self.main_path = main_path
        self.mode = mode
        self.crop_size = crop_size
        self.divider = divider
        self.data_augmentation_times = data_augmentation_times

        self.black_skin_samples = []
        self.clean_samples = []


        self._load_data()

        ## 所有数据的path list
        ## 黑皮label是0， 无黑皮label是1
        self.img_path = self.black_skin_samples + self.clean_samples
        self.labels = [0] * len(self.black_skin_samples) + [1] * len(self.clean_samples)
        self.labels = torch.from_numpy(np.array(self.labels))

    def _load_data(self):
        mode_path = os.path.join(self.main_path, self.mode)
        black_skin_path = os.path.join(mode_path, 'black_skin')
        clean_path = os.path.join(mode_path, 'clean')

        self.black_skin_samples = self._load_samples(black_skin_path)
        self.clean_samples = self._load_samples(clean_path)

    def _load_samples(self, path):
        samples = []
        for sample_dir in os.listdir(path):
            sample_path = os.path.join(path, sample_dir)
            sample_images = os.listdir(sample_path)
            samples.extend([os.path.join(sample_path, img) for img in sample_images])
        return samples

    def _load_and_preprocess(self, path):
        img = Image.open(path)
        size = img.size
        if size[0] % self.divider != 0:
            size_new = (size[0] - size[0] % self.divider, size[1])
        else:
            size_new = size
        joint = Image.new("L", (size_new[0] // self.divider, size_new[1] * self.divider))
        for i in range(self.divider):
            loc = (0, i * size_new[1])
            img_tmp = img.crop((i * (size_new[0] // self.divider), 0, (i + 1) * (size_new[0] // self.divider), size_new[1]))
            joint.paste(img_tmp, loc)
        img = joint.resize(self.crop_size, Image.BICUBIC)
        if self.data_augmentation_times > 0:
                img = self._data_augmentation(img)
        output_img = np.array(img) / 255.0
        # output_img = torch.from_numpy(output_img).to(torch.float32)
        output_img = torchvision.transforms.ToTensor()(output_img)  # 转换为PyTorch张量
        output_img = output_img.float()
        # print(output_img.shape)

        # output_img =  output_img.unsqueeze(0)

        return output_img

    def _data_augmentation(self, img):
        if np.random.randint(0, 2):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = self._enhance_brightness(img)
        img = self._enhance_contrast(img)
        return img

    def _enhance_brightness(self, img):
        enh_bri = ImageEnhance.Brightness(img)
        brightness = np.random.uniform(0.6, 1.6)
        image_brightened = enh_bri.enhance(brightness)
        return image_brightened

    def _enhance_contrast(self, img):
        enh_con = ImageEnhance.Contrast(img)
        contrast = np.random.uniform(0.6, 1.6)
        image_contrasted = enh_con.enhance(contrast)
        return image_contrasted

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self._load_and_preprocess(self.img_path[idx])
        label = self.labels[idx]
        return img, label

def check_dataset(data_loader):
    image, label = iter(data_loader).__next__()
    sample = image[0].squeeze()
    # sample = sample.permute((1,2,0)).numpy()
    plt.imshow(sample)
    plt.show()
    print(label[0].numpy())

if __name__ == '__main__':
    main_path = 'dataset/unwrap_plus'
    crop_size = (224, 224)
    divider = 6
    data_augmentation_times = 3

    train_dataset = BSDataset(main_path, mode='train', crop_size=crop_size, divider=divider,
                                   data_augmentation_times=data_augmentation_times)
    print(train_dataset.__len__())
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    check_dataset(train_loader)


    # for batch in train_loader:
    #     images, labels = batch
    #     # Your training code here
