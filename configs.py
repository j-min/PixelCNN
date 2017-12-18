import argparse
from datetime import datetime
from pathlib import Path
import pprint
import torch

project_dir = Path(__file__).resolve().parent
datasets_dir = project_dir.joinpath('datasets/')

# Where to save checkpoint and log images
result_dir = project_dir.joinpath('results/')
if not result_dir.exists():
    result_dir.mkdir()


def get_optimizer(optimizer_name='Adam'):
    """Get optimizer by name"""
    # optimizer_name = optimizer_name.capitalize()
    return getattr(torch.optim, optimizer_name)


def str2bool(arg):
    """String to boolean"""
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class BaseConfig(object):
    def __init__(self):
        """Base Configuration Class"""

        self.parse_base()

    def parse_base(self):
        """Base configurations for all models"""

        self.parser = argparse.ArgumentParser()

        #================ Mode ==============#
        self.parser.add_argument('--mode', type=str, default='train')

        #================ Train ==============#
        self.parser.add_argument('--batch_size', type=int, default=16)
        self.parser.add_argument('--n_epochs', type=int, default=200)
        self.parser.add_argument('--optimizer', type=str, default='RMSprop')

        self.parser.add_argument('-dataset', type=str, default='CIFAR10')

        #=============== Misc ===============#
        self.parser.add_argument('--log_interval', type=int, default=100)
        self.parser.add_argument('--save_interval', type=int, default=10)

    def parse(self):
        """Update configuration with extra arguments (To be inherited)"""
        pass

    def initialize(self, parse=True, **optional_kwargs):
        """Set kwargs as class attributes with setattr"""

        # Update parser
        self.parse()

        # Parse arguments
        if parse:
            kwargs = self.parser.parse_args()
        else:
            kwargs = self.parser.parse_known_args()[0]

        # namedtuple => dictionary
        kwargs = vars(kwargs)
        kwargs.update(optional_kwargs)

        if kwargs is not None:
            for key, value in kwargs.items():
                if key == 'optimizer':
                    value = get_optimizer(value)
                setattr(self, key, value)

        self.isTrain = self.mode == 'train'

        # Dataset
        # ex) ./datasets/Mnist/
        self.dataset_dir = datasets_dir.joinpath(self.dataset)

        # Save / Log
        # ex) ./results/vae/
        self.model_dir = result_dir
        if not self.model_dir.exists():
            self.model_dir.mkdir()

        if self.mode == 'train':
            # ex) ./results/vae/2017-12-10_10:09:08/
            time_now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.ckpt_dir = self.model_dir.joinpath(time_now)

            self.ckpt_dir.mkdir()
            file_path = self.ckpt_dir.joinpath('config.txt')
            with open(file_path, 'w') as f:
                f.write('------------ Configurations -------------\n')
                for k, v in sorted(self.__dict__.items()):
                    f.write('%s: %s\n' % (str(k), str(v)))
                f.write('----------------- End -------------------\n')

        # Previous ckpt to load (optional for evaluation)
        if self.mode == 'test':
            assert self.load_ckpt_time
            self.ckpt_dir = self.model_dir.joinpath(self.load_ckpt_time)

        return self

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True):
    """Get configuration class in single step"""
    return BaseConfig().initialize(parse=parse)


if __name__ == '__main__':
    config = get_config()
    import ipdb
    ipdb.set_trace()
