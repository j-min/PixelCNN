import time
import numpy as np
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F
import torch
# import torch.cuda as cuda
from torch.autograd import Variable
from torchvision.utils import save_image
from models import PixelCNN


class Solver(object):
    def __init__(self, config, train_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_batches_in_epoch = len(self.train_loader)
        self.total_data_size = len(self.train_loader.dataset)
        self.is_train = self.config.isTrain

    def build(self):
        self.net = PixelCNN().cuda()
        print(self.net, '\n')

        if self.config.mode == 'train':
            self.optimizer = self.config.optimizer(self.net.parameters())
            self.loss_fn = nn.CrossEntropyLoss()

    def train(self):

        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            epoch_i += 1

            # For debugging
            if epoch_i == 1:
                #     self.test(epoch_i)
                self.sample(epoch_i)

            self.net.train()
            self.batch_loss_history = []

            for batch_i, (image, label) in enumerate(tqdm(
                    self.train_loader, desc='Batch', ncols=80, leave=False)):

                batch_i += 1
                # [batch_size, 3, 32, 32]
                image = Variable(image).cuda()

                # [batch_size, 3, 32, 32, 256]
                logit = self.net(image)
                logit = logit.contiguous()
                logit = logit.view(-1, 256)

                target = Variable(image.data.view(-1) * 255).long()

                batch_loss = self.loss_fn(logit, target)

                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

                batch_loss = float(batch_loss.data)
                self.batch_loss_history.append(batch_loss)

                if batch_i > 1 and batch_i % self.config.log_interval == 0:
                    log_string = f'Epoch: {epoch_i} | Batch: ({batch_i}/{self.n_batches_in_epoch}) | '
                    log_string += f'Loss: {batch_loss:.3f}'
                    tqdm.write(log_string)

            epoch_loss = np.mean(self.batch_loss_history)
            tqdm.write(f'Epoch Loss: {epoch_loss:.2f}')

            self.test(epoch_i)
            self.sample(epoch_i)

    def test(self, epoch_i):
        """Compute error on test set"""

        test_errors = []
        # cuda.synchronize()
        start = time.time()

        self.net.eval()

        for image, label in self.test_loader:

            # [batch_size, channel, height, width]
            image = Variable(image.cuda(async=True), volatile=True)

            # [batch_size, channel, height, width, 256]
            logit = self.net(image).contiguous()

            # [batch_size x channel x height x width, 256]
            logit = logit.view(-1, 256)

            # [batch_size x channel x height x width]
            target = Variable((image.data.view(-1) * 255).long())

            loss = F.cross_entropy(logit, target)

            test_error = float(loss.data)
            test_errors.append(test_error)

        # cuda.synchronize()
        time_test = time.time() - start
        log_string = f'Test done! | It took {time_test:.1f}s | '
        log_string += f'Test Loss: {np.mean(test_errors):.2f}'
        tqdm.write(log_string)

    def sample(self, epoch_i):
        """Sampling Images"""

        image_path = str(self.config.ckpt_dir.joinpath(f'epoch-{epoch_i}.png'))
        tqdm.write(f'Saved sampled images at f{image_path})')
        self.net.eval()

        sample = torch.zeros(self.config.batch_size, 3, 32, 32).cuda()

        for i in range(32):
            for j in range(32):

                # [batch_size, channel, height, width, 256]
                out = self.net(Variable(sample, volatile=True))

                # out[:, :, i, j]
                # => [batch_size, channel, 256]
                probs = F.softmax(out[:, :, i, j], dim=2).data

                # Sample single pixel (each channel independently)
                for k in range(3):
                    # 0 ~ 255 => 0 ~ 1
                    pixel = torch.multinomial(probs[:, k], 1).float() / 255.
                    sample[:, k, i, j] = pixel

        import ipdb
        ipdb.set_trace()

        save_image(sample, image_path)
