import torch
import os
from configs import CONFIGS as FLAGS


def save_images(batch, file_order):
    batch_size = batch.shape[0]  # 获取测试的batch_size
    for k in range(batch_size):
        img = batch[k]
        img = img.reshape(FLAGS.dataset.height, FLAGS.dataset.width)
        filename = os.path.join(FLAGS.out_data_dir, '%05d'%file_order + '.pt')
        file_order += 1
        torch.save(img, filename)
