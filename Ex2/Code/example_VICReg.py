from trainer import Trainer
from imports import *
from dataset import RepresentationDataset
from augmentations import test_transform, train_transform

if __name__ == '__main__':
    # NOTES: -1 Passed device='cpu' in Trainer change to 'cuda' if needed

    transform = transforms.Compose([transforms.ToTensor()])
    train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256,
                             shuffle=False)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # loading trainer
    path = './VICReg_model'
    trainer = Trainer(path, train_loader, test_loader, epochs=30, device='cpu',
                      batch_size=256, encoder_dim=128, train_transform=train_transform,
                      test_transform=test_transform, scheduler=None)
    trainer.load_models(linear_classifier=True)

    # Linear probing
    trainer.get_representations(trainset=True)
    trainer.linear_probing(train=False)

    # Retrievals for a sample from each class
    trainer.retrieval_evaluation()
