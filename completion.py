import torch
import os
import time
from torch.utils.tensorboard import SummaryWriter
from model import UNet
from configs import CONFIGS
from tqdm import tqdm
from dataset import CompletionDataset
from torch.utils.data import DataLoader
from utils import schedule, print_info
from utils import save_images
from torchvision import transforms
from torchvision import datasets, transforms


class CompletionSolver():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

        self.start_epoch = 1
        self.model = None           # torch.nn.Module
        self.optimizer = None       # torch.optim.Optimizer
        self.scheduler = None       # torch.optim.lr_scheduler._LRScheduler
        self.summary_writer = None  # torch.utils.tensorboard.SummaryWriter
        self.log_file = None        # str, used to save training logs
        self.ckpt_dir = None
        self.train_loader = None
        self.test_loader = None

    def get_model(self):
        return UNet(self.FLAGS.model.n_channels, self.FLAGS.model.n_classes)
    
    def get_dataset(self, train_test):
        flags = flags = CONFIGS.dataset
        mnist_dataset = datasets.MNIST(flags.root_dir, train= train_test == 'train', download=True, transform=transforms.ToTensor())
        return mnist_dataset
    
    def get_dataloader(self, train_test):
        dataset = self.get_dataset(train_test)
        if train_test == 'train':
            flags = self.FLAGS.train
        elif train_test == 'test':
            flags = self.FLAGS.test
        data_loader = DataLoader(dataset, batch_size=flags.batch_size, shuffle=flags.shuffle)
        return data_loader

    def config_model(self):
        self.model = self.get_model().cuda(self.FLAGS.gpu)

    def config_dataloader(self, train_test):
        if train_test == 'train':
            self.train_loader = self.get_dataloader(train_test)
        elif train_test == 'test':
            self.test_loader = self.get_dataloader(train_test)

    def config_optimizer(self):
        flags = self.FLAGS.optimizer
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(
          parameters, lr=flags.lr, weight_decay=flags.weight_decay)
        
    def config_scheduler(self):
        flags = self.FLAGS.scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                         flags.decay_step, gamma=flags.lr_decay)
    def config_log(self):
        self.logdir = self.FLAGS.logdir
        if not os.path.exists(self.logdir):
            os.mkdir(self.logdir)

        logfile_dir = os.path.join(self.logdir, 'csv')
        if not os.path.exists(logfile_dir):
            os.mkdir(logfile_dir)
        self.log_file = os.path.join(logfile_dir, 'log.csv')    

        summary_writer_dir = os.path.join(self.logdir, 'tensorboard')
        if not os.path.join(summary_writer_dir):
            os.mkdir(summary_writer_dir)
        self.summary_writer = SummaryWriter(summary_writer_dir, flush_secs=20)

        self.ckpt_dir = os.path.join(self.logdir, 'ckpt')
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)

        confidence_writer_dir = os.path.join(self.logdir, 'confidence')
        if not os.path.join(confidence_writer_dir):
            os.mkdir(confidence_writer_dir)
        self.confidence_writer = SummaryWriter(confidence_writer_dir, flush_secs=20)

    def save_checkpoint(self, epoch):
        # clean up
        ckpts = sorted(os.listdir(self.ckpt_dir))
        ckpts = [ck for ck in ckpts if ck.endswith('.pth') or ck.endswith('.tar')]
        if len(ckpts) > self.FLAGS.ckpt_num:
            for ckpt in ckpts[:-self.FLAGS.ckpt_num]:
                os.remove(os.path.join(self.ckpt_dir, ckpt))

        # save ckpt
        model_dict = self.model.state_dict()
        ckpt_name = os.path.join(self.ckpt_dir, '%05d' % epoch)
        torch.save(model_dict, ckpt_name + '.model.pth')
        torch.save({'model_dict': model_dict, 'epoch': epoch,
                    'optimizer_dict': self.optimizer.state_dict(),
                    'scheduler_dict': self.scheduler.state_dict(), },
                    ckpt_name + '.solver.tar')

    def load_checkpoint(self):
        ckpt = self.FLAGS.ckpt
        if not ckpt:
        # If ckpt is empty, then get the latest checkpoint from ckpt_dir
            if not os.path.exists(self.ckpt_dir):
                return
            ckpts = sorted(os.listdir(self.ckpt_dir))
            ckpts = [ck for ck in ckpts if ck.endswith('solver.tar')]
            if len(ckpts) > 0:
                ckpt = os.path.join(self.ckpt_dir, ckpts[-1])
        if not ckpt:
            return  # return if ckpt is still empty

        # load trained model
        # check: map_location = {'cuda:0' : 'cuda:%d' % self.rank}
        trained_dict = torch.load(ckpt, map_location='cuda:%d' % self.FLAGS.gpu)
        if ckpt.endswith('.solver.tar'):
            model_dict = trained_dict['model_dict']
            self.start_epoch = trained_dict['epoch'] + 1  # !!! add 1
            if self.optimizer:
                self.optimizer.load_state_dict(trained_dict['optimizer_dict'])
            if self.scheduler:
                self.scheduler.load_state_dict(trained_dict['scheduler_dict'])
        else:
            model_dict = trained_dict
        self.model.load_state_dict(model_dict)
        if self.FLAGS.run == 'train':
            print(f'Start from epoch {self.start_epoch}')
        else:
            print(f'Loading checkpoint at epoch {self.start_epoch-1}')

    def compute_loss(self, logits, target, mask):
        logits = logits.reshape(-1)  # size:[]
        target = target.reshape(-1)  # size: batch_size
        # loss = torch.nn.functional.cross_entropy(logits[mask], target[mask])
        loss = torch.nn.functional.binary_cross_entropy(logits[mask], target[mask].float())
        return loss
    
    def compute_batch_acc(self, final_indices, target):
        final_indices = final_indices.reshape(-1)
        target = target.reshape(-1)

        final_indices = final_indices.round()
        correct = final_indices.eq(target).float().mean()

        return correct

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_batch_acc = 0
        loop_train = tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=80)
        loop_train.set_description(f'Epoch [{epoch}/{self.FLAGS.train.max_epoch}]')
        for _, data in loop_train:
            self.optimizer.zero_grad()
            img = data[0].cuda(self.FLAGS.gpu)
            b, c, h, w = img.shape
            img = img.reshape(-1)
            img = img.round()
            label = img.clone()  #.type(torch.int64)
            # 采样一个比例随机进行 mask
            ratio = torch.rand(1)
            mask = torch.randperm(b * h *w, device=img.device)
            mask = mask[: int(ratio * b * h * w)]
            img[mask] = -1
            img = img.reshape(b, c, h, w)
            # 训练
            out = self.model(img)
            loss = self.compute_loss(out, label, mask)
            loss.backward()
            self.optimizer.step()
            total_loss += loss
            out = out.reshape(-1)
            out = out.round()
            total_batch_acc += self.compute_batch_acc(out, label)
        return total_loss / len(self.train_loader), total_batch_acc / (len(self.train_loader))

    def test_epoch(self):
        self.model.eval()
        total_loss = 0
        total_batch_acc = 0
        loop_test = tqdm(enumerate(self.test_loader), total=len(self.test_loader), ncols=80)
        loop_test.set_description(f'Testing......')
        for _, data in loop_test:
            self.optimizer.zero_grad()
            img = data[0].cuda(self.FLAGS.gpu)
            b, c, h, w = img.shape
            img = img.reshape(-1)
            img = img.round()
            label = img.clone()
            # 采样一个比例随机进行 mask
            ratio = torch.rand(1)
            mask = torch.randperm(b * h *w, device=img.device)
            mask = mask[: int(ratio * b * h * w)]
            img[mask] = -1
            img = img.reshape(b, c, h, w)
            # 训练
            out = self.model(img)
            loss = self.compute_loss(out, label, mask)
            loss.backward()
            self.optimizer.step()
            total_loss += loss
            out = out.reshape(-1)
            out = out.round()
            total_batch_acc += self.compute_batch_acc(out, label)
        return total_loss / len(self.test_loader), total_batch_acc / (len(self.test_loader))
    
    def eval_step(self, cur_ids, step, unknown_number_in_the_beginning, choice_temperature=1.0, 
           mask_scheduling_method="cosine"):
        '''
        cur_ids: [batch_size, channels, height, width] input masked image,
        tokens_to_logits: decoder function taking single token slices and cache and
        returning logits and updated cache.
        choice_temperature: float: temperature to control the randomness of masking.
        mask_scheduling_method: masking method string. See mask_schedule.py for
        details.

        return: sampled_ids:带有mask的[batch, h*w]
               final_ids: 最终的图像 [batch, h*w]
               logits: [batch_size, h*w, 2] 每个像素位置对应的概率
        '''
        # 使用函数token_to_logits预测logits
        logits = self.model(cur_ids)
        # 根据logits采样 index, [batch_size, h * w]
        sampled_ids, selected_probs = self.sample_logit(logits)
        # 获取输入 indices 中 mask 的蒙版
        cur_ids = cur_ids.reshape(cur_ids.shape[0], -1)
        unknown_map = (cur_ids == self.FLAGS.dataset.mask_id)
        # 更新masked token, 原来不被 mask 的位置，还是选择原来的token_ids，原来被mask的位置选择新的index
        final_ids = torch.where(unknown_map, sampled_ids, cur_ids)  # sampled_ids 就是最该迭代预测出的图像
        # Ignores the tokens given in the input by overwriting their confidence.
        selected_probs = torch.where(unknown_map, selected_probs, 1.0e6)  # selected_probs是预测出的最终概率
        # Defines the mask ratio for the next round. The number to mask out is
        # determined by mask_ratio * unknown_number_in_the_beginning.
        ratio = 1. * (step + 1) / self.FLAGS.test.num_iter
        mask_ratio = schedule(ratio, unknown_number_in_the_beginning, mask_scheduling_method)
        # Gets mask lens for each sample in the batch according to the mask ratio.
        mask_len = torch.floor(unknown_number_in_the_beginning * mask_ratio).unsqueeze(1)
        # Keeps at least one of prediction in this round and also masks out at least
        # one and for the next iteration
        mask_len = torch.maximum(
            torch.tensor(1),
            torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True) - 1, mask_len))
        # Adds noise for randomness
        masking = self.mask_by_noise(mask_len, selected_probs, choice_temperature * (1. - ratio))
        # Masks tokens with lower confidence. 最终得到这个最后的 indices，形状为[batch_size, w*h]
        sampled_ids = torch.where(masking, self.FLAGS.dataset.mask_id, final_ids)
        check = torch.where(unknown_map, logits, sampled_ids)
        self.visualize_in_2d(masking[0], step, name_prefix='masking')
        self.visualize_in_2d(logits[0], step, name_prefix='logits')
        self.visualize_in_2d(check[0], step, name_prefix='logits_updated')
        self.visualize_in_2d(selected_probs[0].clamp(0, 1), step, name_prefix='confidence')
        return sampled_ids, final_ids, logits, selected_probs

    def mask_by_noise(self, mask_len, probs, temperature=1.0):
        """
        mask_len为要mask像素的数量
        probs为采样出来的概率
        temperature控制随机性

        return: 下次迭代要mask的蒙版
        """
        confidence = probs

        # 下面的指令为增加随机性
        confidence = torch.log(confidence) + temperature * torch.distributions.gumbel.Gumbel(torch.tensor([1.0]), torch.tensor([2.0])).sample(confidence.shape).squeeze(-1).cuda(self.FLAGS.gpu)
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        cut_off = torch.gather(sorted_confidence, dim=-1, index=mask_len.type(torch.int64))
        masking = (confidence < cut_off)
        return masking

    def sample_logit(self, prob):
        '''
        prob: sigmoid后的概率, 目前仅是对于单个数据的
        return: 采样出的indices和对应的概率
        '''
        batch_size = prob.shape[0]
        prob = prob.reshape(-1, 1)
        prob = torch.concat([1-prob, prob], dim=1)
        indices = torch.multinomial(prob, 1)
        prob = torch.gather(prob, dim=1, index=indices)
        return indices.reshape(batch_size, -1), prob.reshape(batch_size, -1)

    def sample_logit_(self, prob):
        '''
        prob: softmax后的概率, 目前仅是对于单个数据的
        return: 采样出的indices [batch_size, h*w] 和 对应的概率 [batch_size, h*w]
        '''
        batch_size = prob.shape[0]
        prob = prob.reshape(-1, prob.shape[-1])
        prob = torch.nn.functional.softmax(prob, dim=-1)
        indices = torch.multinomial(prob, 1)
        prob = torch.gather(prob, dim=1, index=indices)
        return indices.reshape(batch_size, -1), prob.reshape(batch_size, -1)
    
    def train(self):
        self.config_model()
        self.config_dataloader('train')
        self.config_dataloader('test')
        self.config_optimizer()
        self.config_scheduler()
        self.config_log()
        self.load_checkpoint()

        for epoch in range(self.start_epoch, self.FLAGS.train.max_epoch + 1):
            # 训练
            start = time.time()
            train_loss, train_acc = self.train_epoch(epoch)
            self.summary_writer.add_scalar('train_loss', train_loss, epoch)
            self.summary_writer.add_scalar('train_acc', train_acc, epoch)
            self.scheduler.step()

            # 测试
            test_loss, test_acc = self.test_epoch()
            self.summary_writer.add_scalar('test_loss', test_loss, global_step=epoch)
            self.summary_writer.add_scalar('test_acc', test_acc, global_step=epoch)
            duration = time.time() - start
            info = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'current_epoch': epoch,
                'epochs': self.FLAGS.train.max_epoch,
                't_duration': duration
            }
            print_info(info)

            if epoch % 100 == 0:
                self.save_checkpoint(epoch)

    def eval(self):
        self.config_model()
        self.config_dataloader('test')
        self.config_log()
        self.load_checkpoint()
        
        self.model.eval()
        loop_test = tqdm(range(10), ncols=80)
        loop_test.set_description('Evaluating......')
        with torch.no_grad():
            file_order = 0  # 输出的文件序号，用于给文明命名
            for i in loop_test:
                img = -torch.ones((self.FLAGS.test.batch_size, 1, 28, 28)).cuda(self.FLAGS.gpu)
                 # 开始时所有mask_tokens的数量, cur_ids为输入的图像数据
                cur_ids_seq = img.reshape(img.shape[0], -1)
                unknown_number_in_the_beginning = torch.sum(cur_ids_seq == self.FLAGS.dataset.mask_id, axis=-1)
                cur_ids = img
                for step in range(self.FLAGS.test.num_iter):
                    cur_ids, final_ids, logits, confidence = self.eval_step(cur_ids, step, unknown_number_in_the_beginning,
                                              choice_temperature=1.0, mask_scheduling_method="cosine")
                    
                    cur_ids = cur_ids.reshape(cur_ids.shape[0], 1, 
                                              self.FLAGS.dataset.height, self.FLAGS.dataset.width)

                    self.confidence_writer.add_histogram(tag='logits_histogram', values=logits.reshape(-1).clamp(0, 1), global_step=step)
                    self.confidence_writer.add_histogram(tag='confidence_histogram', values=confidence.reshape(-1).clamp(0, 1), global_step=step)
                # save_images(final_ids, file_order)  
                batch_size = final_ids.shape[0]
                final_ids = final_ids.reshape(batch_size, self.FLAGS.dataset.height, self.FLAGS.dataset.width)
                toPIL = transforms.ToPILImage()
                out_data_dir = os.path.join(self.logdir, 'out_data')
                if not os.path.exists(out_data_dir):
                    os.makedirs(out_data_dir)
                for k in range(batch_size):
                    image = final_ids[k]
                    pic = toPIL(image)
                    filename = os.path.join(out_data_dir, '%05d'%file_order + '.jpg')
                    file_order += 1
                    pic.save(filename)   

    def eval_old(self):
        self.config_model()
        self.config_dataloader('test')
        self.config_log()
        self.load_checkpoint()
        
        self.model.eval()
        total_loss = [0 for _ in range(self.FLAGS.test.num_iter)]
        total_batch_acc = [0 for _ in range(self.FLAGS.test.num_iter)]
        loop_test = tqdm(enumerate(self.test_loader), total=len(self.test_loader), ncols=80)
        loop_test.set_description('Evaluating......')
        with torch.no_grad():
            file_order = 0  # 输出的文件序号，用于给文明命名
            for _, data in loop_test:
                data['x'], data['y'] = data['x'].cuda(self.FLAGS.gpu), data['y'].cuda(self.FLAGS.gpu)
                 # 开始时所有mask_tokens的数量, cur_ids为输入的图像数据
                cur_ids_seq = data['x'].reshape(data['x'].shape[0], -1)
                unknown_number_in_the_beginning = torch.sum(cur_ids_seq == self.FLAGS.dataset.mask_id, axis=-1)
                cur_ids = data['x']
                for step in range(self.FLAGS.test.num_iter):
                    cur_ids, final_ids, logits = self.eval_step(cur_ids, step, unknown_number_in_the_beginning,
                                              choice_temperature=1.0, mask_scheduling_method="cosine")
                    
                    # 在self.test_step函数内部计算整正确率和loss
                    cur_ids = cur_ids.reshape(cur_ids.shape[0], 1, 
                                              self.FLAGS.dataset.height, self.FLAGS.dataset.width)
                    total_loss[step] += (self.compute_loss(logits, data['y']) / len(self.test_loader)).item()
                    total_batch_acc[step] += self.compute_batch_acc(final_ids, data['y']) / len(self.test_loader)

                # save_images(final_ids, file_order)  
                batch_size = final_ids.shape[0]
                final_ids = final_ids.reshape(batch_size, self.FLAGS.dataset.height, self.FLAGS.dataset.width)
                toPIL = transforms.ToPILImage()
                if not os.path.exists(self.FLAGS.out_data_dir):
                    os.makedirs(self.FLAGS.out_data_dir)
                for k in range(batch_size):
                    image = final_ids[k]
                    pic = toPIL(image)
                    filename = os.path.join(self.FLAGS.out_data_dir, '%05d'%file_order + '.jpg')
                    file_order += 1
                    pic.save(filename)
                    
            for i in range(self.FLAGS.test.num_iter):
                print('loss: ', total_loss[i], 'acc: ', total_batch_acc[i])
                self.summary_writer.add_scalar(f'eval_loss', total_loss[i], i)
                self.summary_writer.add_scalar(f'eval_acc', total_batch_acc[i], i)

    def visualize_in_2d(self, array, step, name_prefix='confidence'):
            '''
            可视化logits. logits的形状为[h, w]
            '''
            array = array.reshape(1, 28, 28)
            # 生成一个图像，并放入tensorboard
            self.summary_writer.add_image(f'{name_prefix}_{step}', array, global_step=step)

    def run(self):
        eval('self.%s()' % self.FLAGS.run)

    @classmethod
    def main(cls):
        completion = cls(CONFIGS)
        completion.run()

if __name__ == '__main__':
    CompletionSolver.main()
