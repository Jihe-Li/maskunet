import os
import torch
from torchvision import datasets, transforms
from configs import CONFIGS
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs import args

def download_mnist(flags):
    if not os.path.exists(flags.root_dir):
        os.mkdir(flags.root_dir)
    print('----------downloading......----------')
    mnist_train = datasets.MNIST(flags.root_dir, train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(flags.root_dir, train=False, download=True, transform=transforms.ToTensor())
    print('----------downloaded!----------')
    return mnist_train, mnist_test

def process_dataset(dataset, train_test, flags):
    # dataset：训练或测试的数据集
    # train_test：'train' 或者 'test'
    # flags: FLAGS.DATA.dataset
    data_loader = DataLoader(dataset, batch_size=1)  # 创建数据加载器
    dir = flags.data_process_dir + '/' + train_test  # 训练或测试数据保存的路径
    if not os.path.exists(dir):
        os.makedirs(dir)

    # 若数据处理完，则不进行再次处理
    folder_labels = os.listdir(dir)  
    folder_images = os.listdir(dir)
    if len(folder_labels) == len(data_loader) and len(folder_images) == len(data_loader) and not args.process_dataset:
        print(f'{train_test} dataset is ready!')
        return

    for i, data in tqdm(enumerate(data_loader), total = len(data_loader), ncols=80):
        img = data[0][0][0]  # [1, height, width]
        filename = dir + '/' + '%05d' % i + '.pt'
        if train_test == 'test':
            save_test_data(img, filename, flags)
        elif train_test == 'train':
            save_train_data(img, filename, flags)

def save_train_data(img, filename, flags):
    img = img.round()
    label = img.type(torch.int64)
#    img_shape = list(img.shape)  # 获取图像原始的形状
#    # 将白色的点设置为 flags.white, 黑色的点设置为flags.black
#    img = img.ceil()
    # 随机选择部分点进行mask
#    choice = torch.randint(high=min(img_shape[0], img_shape[1]), 
#                size=(2, int(flags.ratio * img_shape[0] * img_shape[1])))
#    choice = (choice[0], choice[1])
#    img[choice] = flags.mask_id
    data = {'x': img.unsqueeze(0), 'y': label}
    torch.save(data, filename)

def save_test_data(img, filename, flags):
    label = img.round()  # 存储标签
    label = label.type(torch.int64)
    mask = label == 0  
    # 更新mask，使一些原本为1的像素变为0 
    indices = torch.where(mask == False)  # (-2维的索引， -1维的索引)
    length = len(indices[0])
    choice = torch.randint(high=length, size=(int(flags.ratio*length),))
    indices_rm = []  # 创建一个空列表用于存储被随机剔除的indices
    for i in indices:
        i = i[choice]
        indices_rm.append(i)
    indices_rm = tuple(indices_rm)
    mask[indices_rm] = True
    img[~mask], img[mask] = flags.white_id, flags.mask_id
    data = {'x': img.unsqueeze(0), 'y': label}  # 将图像数据和标签存储为一个字典
    torch.save(data, filename)

if __name__ == '__main__':
    flags = CONFIGS.dataset
    train_dataset, test_dataset = download_mnist(flags)
    process_dataset(train_dataset, 'train', flags)  # 处理训练集数据
    process_dataset(test_dataset, 'test', flags)  # 处理测试机数据
