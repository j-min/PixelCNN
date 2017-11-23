from tqdm import tqdm, trange
import torch.optim as optim
from models import PixelCNN
from torch.autograd import Variable

class Solver(object):
    def __init__(self, config, train_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

    def build(self):
        self.net = PixelCNN(
            self.config.c_in,
            self.config.dim,
            self.config.c_out,
            self.config.k_size,
            self.config.stride,
            self.config.pad).cuda()

        if self.config.mode == 'train':
            self.net.train()
            self.optimizer = optim.Adam(self.net.parameters())

    def train(self):
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            for batch_i, (image, label) in enumerate(tqdm(
                self.train_loader, desc='Batch', ncols=80, leave=False)):

                # image: [batch_size, 3, 32, 32]
                image = Variable(image).cuda()
