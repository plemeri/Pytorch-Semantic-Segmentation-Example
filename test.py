import argparse
import datetime
import time

from torch.utils.data import DataLoader

from dataset import *
from deeplab import *


class Test:
    def __init__(self):
        self.FLAGS = self.args()

        # dataset
        self.dataset = PennFudanSegmentation(split='test')
        self.dataloader = DataLoader(self.dataset, 1, True, num_workers=4, drop_last=False)

        # model
        self.model = DeepLab(self.FLAGS.backbone, self.FLAGS.class_num, self.FLAGS.stride)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in self.FLAGS.device_ids])
        if len(self.FLAGS.device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(self.FLAGS.device_ids))])
        self.model = self.model.cuda()

        # load checkpoint if exist
        self.load_checkpoint()

        if os.path.isdir(self.FLAGS.result_dir) is False:
            os.makedirs(self.FLAGS.result_dir)

        # metrics (accuracy, mean iou)
        self.metrics = Metrics(self.FLAGS.class_num, self.FLAGS.ignore_mask)
        self.global_step = 1
        self.cur_time = time.time()

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

    def eval(self):
        self.metrics.clear_confusion_matrix()
        self.model.eval()
        for i, batch in enumerate(self.dataloader):
            image = batch['image'].cuda()
            label = batch['label'].cuda()

            out = self.model(image)

            log = self.metrics(out, label, mode='test')
            self.logger(log, self.global_step)

            prediction = out.squeeze().argmax(axis=0).cpu().numpy().astype(np.uint8) * 255
            prediction = np.stack([prediction] * 3, axis=-1)
            label = label.squeeze().cpu().numpy().astype(np.uint8) * 255
            label = np.stack([label] * 3, axis=-1)
            image = image.squeeze().permute(1, 2, 0).cpu().numpy()
            image *= [0.229, 0.224, 0.225]
            image += [0.485, 0.456, 0.406]
            image *= 255
            image = image.astype(np.uint8)

            Image.fromarray(np.vstack([image, label, prediction])).save(os.path.join(self.FLAGS.result_dir, str(i) + '.png'))

            self.global_step += 1

    def args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--backbone', type=str, default='ResNet50', choices=['ResNet18', 'ResNet50', 'ResNet101', 'ResNet152'])
        parser.add_argument('--class_num', type=int, default=2)
        parser.add_argument('--stride', type=int, default=16, choices=[8, 16])
        parser.add_argument('--device_ids', type=int, nargs='+', default=[0])
        parser.add_argument('--ignore_mask', type=int, default=255)
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_pedcut')
        parser.add_argument('--result_dir', type=str, default='./results2')
        return parser.parse_args()


if __name__ == "__main__":
    tester = Test()
    tester.eval()
