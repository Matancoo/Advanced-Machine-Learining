from imports import *


# Datasets
class RepresentationDataset(Dataset):
    def __init__(self, representations, labels):
        self.representations = representations
        self.labels = labels

    def __len__(self):
        # Dataset length
        return len(self.representations)

    def get_represenatation(self, idx):
        return self.representations[idx]

    def __getitem__(self, idx):
        # Get item by index
        return self.representations[idx], self.labels[idx]


class TestDataset(Dataset):
    def __init__(self):
        # Transforms for CIFAR-10
        transform_cifar = transforms.Compose([transforms.ToTensor()])

        # Transforms for MNIST
        transform_mnist = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])

        # Load the MNIST test dataset
        mnist_testset = datasets.MNIST(root='./', train=False, download=True, transform=transform_mnist)

        self.mnist_data = np.array([img.numpy() for img, _ in mnist_testset])
        self.mnist_labels = np.ones(len(mnist_testset))

        # Load the CIFAR-10 test dataset
        cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
        self.cifar_data = np.array(cifar_testset.data)
        self.cifar_data = self.cifar_data.reshape(self.mnist_data.shape)
        self.cifar_labels = np.zeros(len(cifar_testset))

        # Concatenate the data and labels
        self.data = np.concatenate((self.cifar_data, self.mnist_data), axis=0)
        self.labels = np.concatenate((self.cifar_labels, self.mnist_labels), axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_x_y(self):
        return self.data, self.labels
