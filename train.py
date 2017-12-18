from configs import BaseConfig as Config
from data_loader import get_loader
from solver import Solver


def main(config):
    train_loader = get_loader(
        config.dataset_dir,
        config.batch_size,
        train=True)
    test_loader = get_loader(
        config.dataset_dir,
        config.batch_size,
        train=False)

    solver = Solver(config, train_loader=train_loader, test_loader=test_loader)
    print(config)
    # print(f'\nTotal data size: {solver.total_data_size}\n')

    solver.build()
    solver.train()


if __name__ == '__main__':
    # Get Configuration
    config = Config().initialize()
    # import ipdb
    # ipdb.set_trace()
    main(config)
