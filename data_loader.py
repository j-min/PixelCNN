from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor()
    ])

def get_loader(directory='./data', batch_size=128, train=True,
    num_workers=1, pin_memory=True):
    shuffle = not train
    # 32 x 32
    dataset = datasets.CIFAR10(
        directory, batch_size=batch_size,
        train=train, shuffle=shuffle, download=True,
        transform=transform)
    loader = DataLoader(dataset, num_workers=num_workers, pin_memory=False)
    return loader
