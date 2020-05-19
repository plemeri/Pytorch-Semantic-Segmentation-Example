import argparse
import datetime
import time

from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from dataset import *
from deeplab import *


class PolyLr(_LRScheduler):
    def __init__(self, optimizer, gamma, max_iteration, warmup_iteration=0, last_epoch=-1):
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.warmup_iteration = warmup_iteration
        super(PolyLr, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iteration:
            alpha = self.last_epoch / self.warmup_iteration

            return [min(base_lr * (1 / 10.0 * (1 - alpha) + alpha),
                        base_lr * ((1 - (self.last_epoch / self.max_iteration)) ** (self.gamma))) for base_lr in
                    self.base_lrs]

        else:
            return [base_lr * ((1 - (self.last_epoch / self.max_iteration)) ** self.gamma) for base_lr in self.base_lrs]


class Train:
    def __init__(self):
        self.FLAGS = self.args()

        # dataset
        self.dataset = PedestrianSegmentation(split='train')
        self.dataloader = DataLoader(self.dataset, self.FLAGS.batch_size, True, num_workers=4, drop_last=True)

        # model
        self.model = DeepLab(self.FLAGS.backbone, self.FLAGS.class_num, self.FLAGS.stride)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in self.FLAGS.device_ids])
        if len(self.FLAGS.device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(self.FLAGS.device_ids))])
        self.model = self.model.cuda()

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         self.FLAGS.learning_rate * len(self.FLAGS.device_ids),
                                         momentum=0.9, weight_decay=5e-4, nesterov=True)

        # scheduler
        self.scheduler = PolyLr(self.optimizer, 0.9,
                                self.FLAGS.epochs * (len(self.dataset) // self.FLAGS.batch_size) + 1)

        # criterion (loss)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.FLAGS.ignore_mask)
        self.criterion = self.criterion.cuda()

        # metrics (accuracy, mean iou)
        self.metrics = Metrics(self.FLAGS.class_num, self.FLAGS.ignore_mask)
        self.global_step = 1
        self.last_epoch = 0
        self.cur_time = time.time()

        # load checkpoint if exist
        self.load_checkpoint()

    def save_checkpoint(self, tag, epoch):
        print('Snapshot Checkpoint...')
        if len(self.FLAGS.device_ids) > 1:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        torch.save({
            'last_epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, os.path.join(self.FLAGS.checkpoint_dir, 'checkpoint_' + str(tag) + '.pth'))

    def load_checkpoint(self):
        # checkpoint
        if os.path.isdir(self.FLAGS.checkpoint_dir) is False:
            os.makedirs(self.FLAGS.checkpoint_dir)
            return

        files = os.listdir(self.FLAGS.checkpoint_dir)
        files.sort(key=len)

        if len(files) == 0:
            return

        load_dir = os.path.join(self.FLAGS.checkpoint_dir, files[-1])
        if os.path.isfile(load_dir) is True:
            loaded_state_dict = torch.load(load_dir, map_location='cpu')
            print("Checkpoint loaded from " + load_dir)

            self.last_epoch = loaded_state_dict['last_epoch']
            self.global_step = (len(self.dataset) // self.FLAGS.batch_size) * self.last_epoch + 1
            self.optimizer.load_state_dict(loaded_state_dict['optimizer_state_dict'])
            self.scheduler.load_state_dict(loaded_state_dict['scheduler_state_dict'])

            try:
                self.model.load_state_dict(loaded_state_dict['model_state_dict'])
            except RuntimeError:
                self.model.module.load_state_dict(loaded_state_dict['model_state_dict'])

        else:
            print("Checkpoint load failed")

    @staticmethod
    def logger(log_dict, step, delim=' | '):
        log = datetime.datetime.now().isoformat() + delim
        log += 'step [' + str(step) + '] ' + delim
        for key in log_dict.keys():
            log += key + ': {:0.4f}'.format(log_dict[key]) + delim
        print(log)

    def train_epoch(self):
        self.metrics.clear_confusion_matrix()
        self.model.train()
        for batch in self.dataloader:
            image = batch['image'].cuda()
            label = batch['label'].cuda()

            out = self.model(image)
            loss = self.criterion(out, label.long())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            log = self.metrics(out, label)
            log.update({'loss': loss.item()})
            self.logger(log, self.global_step)
            self.global_step += 1

    def fit(self):
        for epoch in range(self.last_epoch + 1, self.FLAGS.epochs + 1):
            self.train_epoch()
            if epoch % self.FLAGS.save_per_epoch == 0:
                self.save_checkpoint(self.global_step, epoch)
        self.save_checkpoint('final', self.FLAGS.epochs + 1)

    def args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--backbone', type=str, default='ResNet50')
        parser.add_argument('--class_num', type=int, default=2)
        parser.add_argument('--stride', type=int, default=16)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--learning_rate', type=float, default=0.1)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--device_ids', type=int, nargs='+', default=[0, 1])
        parser.add_argument('--ignore_mask', type=int, default=255)
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
        parser.add_argument('--save_per_epoch', type=int, default=5)
        return parser.parse_args()


if __name__ == "__main__":
    trainer = Train()
    trainer.fit()
