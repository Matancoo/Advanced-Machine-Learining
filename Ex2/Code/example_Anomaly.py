from imports import *
from dataset import TestDataset
from trainer import Trainer
from helpers import compute_density,plot_roc


if __name__ == '__main__':
    # NOTES: - Passed device='cpu' in Trainer change to 'cuda' if needed

    # Dataset for Anaomaly Detection (AD):
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = TestDataset()
    train_set = datasets.CIFAR10(root='./data', train=True,
                                 download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,num_workers=2)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2)  # removed shuffle for AD

    # Define Trainer for VICReg and NGN methods
    path = './VICReg_model'
    trainer = Trainer(path, train_loader, test_loader, epochs=30, device='cpu',
                      batch_size=256, encoder_dim=128)
    # Load models
    trainer.load_models()
    # Compute density scores
    score_vic = compute_density(trainer, k=2)
    # Plot roc
    plot_roc(test_set.labels, score_vic, 'ROC Curve: VICReg:')
