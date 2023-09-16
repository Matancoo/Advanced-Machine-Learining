from imports import *
from models import Encoder, Projector


class Trainer(object):
    def __init__(self, path,
                 train_loader, test_loader, epochs=10, device='gpu',
                 batch_size=256, encoder_dim=128, scheduler=None, train_transform=None,
                 test_transform=None):
        self.path = path
        self.encoder = Encoder(D=encoder_dim, device=device)  # Encoder
        self.projector = Projector(self.encoder.D, proj_dim=512).to(device)  # Projection head
        self.T = self.transform_batch  # stochastic augmentations on batch
        self.linear_classifier = None
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.representations = []
        self.representations_labels = []
        self.linear_probing_acc = 0
        self.neighboring_indices = []

        self.params = list(self.encoder.parameters()) + list(self.projector.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-6)
        self.scheduler = scheduler

        # Tranformation functions
        self.train_transform = train_transform
        self.test_transform = test_transform

        # Hyperparameters for loss functions
        self.lambd = 25.
        self.eps = 1e-4
        self.nu = 1.

        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        # losses: A dictionary to store all three types of losses
        self.train_losses = {'var': [], 'inv': [], 'cov': [], 'total_vic': []}
        self.validation_losses = {'var': [], 'inv': [], 'cov': [], 'total_vic': []}

    def transform_batch(self, data, train=True):
        """
        Transform the data using the appropriate transformations for training or testing.

        Returns:
            torch.Tensor: The transformed data.
        """
        transform = self.train_transform if train else self.test_transform
        data_view = torch.stack(list(map(transform, data)))
        return data_view

    def save_models(self, encoder=True, projector=True, losses=True, linear_classifier=False):
        if encoder:
            torch.save(self.encoder.state_dict(), self.path + '/encoder.pth')
        if projector:
            torch.save(self.projector.state_dict(), self.path + '/projector.pth')
        if losses:
            torch.save(self.train_losses, self.path + '/train_losses.pt')
            torch.save(self.validation_losses, self.path + '/validation_losses.pt')
        if linear_classifier:
            torch.save(self.linear_classifier, self.path + '/linear_classifier.pth')

    def load_models(self, encoder=True, projector=True, losses=True, linear_classifier=False):
        # map_location for CPU only machines
        if encoder:
            self.encoder.load_state_dict(torch.load(self.path + '/encoder.pth',map_location=torch.device('cpu')))
        if projector:
            self.projector.load_state_dict(torch.load(self.path + '/projector.pth',map_location=torch.device('cpu')))
        if losses:
            self.train_losses = torch.load(self.path + '/train_losses.pt',map_location=torch.device('cpu'))
            # self.validation_losses = torch.load(self.path + '/validation_losses.pt',map_location=torch.device('cpu'))
        if linear_classifier:
            self.linear_classifier = torch.load(self.path + '/linear_classifier.pth',map_location=torch.device('cpu'))

    def eval_VICReg(self):
        self.encoder.eval()
        self.projector.eval()
        for (data, labels) in self.test_loader:
            z1 = self.projector(self.encoder(self.T(data).to(self.device)))  # Z1 = projections represntation 1
            z2 = self.projector(
                self.encoder(self.T(data).to(self.device)))  # Z2 = projections from same augmented batch B

            # VICReg Loss
            inv_loss, var_loss, cov_loss, vic_loss = self.loss_VICReg(z1, z2)

            # Update losses dictionary
            self.validation_losses['var'].append(var_loss.item())
            self.validation_losses['inv'].append(inv_loss.item())
            self.validation_losses['cov'].append(cov_loss.item())
            self.validation_losses['total_vic'].append(vic_loss.item())

    def train_VICReg(self, mu=25.):
        for epoch in range(self.epochs):
            self.encoder.train()
            self.projector.train()
            for b_idx, (data, labels) in enumerate(self.train_loader):
                # Transform image
                z1 = self.projector(self.encoder((self.T(data).to(self.device))))
                z2 = self.projector(self.encoder((self.T(data).to(self.device))))
                # VICReg Loss
                inv_loss, var_loss, cov_loss, vic_loss = self.loss_VICReg(z1, z2, mu=mu)

                # Backward on projector and encoder
                vic_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update losses dictionary
                self.train_losses['var'].append(var_loss.item())
                self.train_losses['inv'].append(inv_loss.item())
                self.train_losses['cov'].append(cov_loss.item())
                self.train_losses['total_vic'].append(vic_loss.item())

                # printing:
                if b_idx % 100 == 0:
                    print("losses:\n", "var_loss", var_loss.item(), inv_loss.item(), "cov_loss:", cov_loss.item())
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, b_idx * len(data), len(self.train_loader.dataset),
                               100. * b_idx / len(self.train_loader), vic_loss.item()))
            self.eval_VICReg()

    def train_VICReg_no_generation(self, mu=25.):
        self.encoder.train()
        self.projector.train()
        self.get_nearest_neighbors(3, self.representations)

        # run for 1 epoch
        for b_idx, (data, labels) in enumerate(self.train_loader):
            batch_size = data.shape[0]

            # get nearest neigbor
            neighbor_indices = self.neighboring_indices[b_idx * batch_size:(b_idx + 1) * batch_size]
            nearest_nbrs_idx = [np.random.choice(indices) for indices in neighbor_indices]
            nearest_nbrs_repr = torch.tensor(self.representations[nearest_nbrs_idx], device=self.device,
                                             dtype=torch.float32)

            z1 = self.projector(nearest_nbrs_repr)
            z2 = self.projector(self.encoder((data.to(self.device))))

            # VICReg Loss
            inv_loss, var_loss, cov_loss, vic_loss = self.loss_VICReg(z1, z2, mu=mu)

            # Backward on projector and encoder
            vic_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update losses dictionary
            self.train_losses['var'].append(var_loss.item())
            self.train_losses['inv'].append(inv_loss.item())
            self.train_losses['cov'].append(cov_loss.item())
            self.train_losses['total_vic'].append(vic_loss.item())

    def get_nearest_neighbors(self, number_of_neighbors, representations):

        nbrs = NearestNeighbors(n_neighbors=number_of_neighbors + 1, algorithm='auto').fit(representations)
        distances, indices = nbrs.kneighbors(representations)
        self.neighboring_indices = indices[:, 1:]  # We start from 1 to exclude self

    # Direct implementation
    # for repr in self.representations:
    #     # Compute Euclidean distances to all other representations
    #     distances = list(map(lambda x: np.linalg.norm(repr - x), self.representations))
    #     # Get indices of nearest neighbors
    #     nearest_neighbors = np.argsort(distances)[1:number_of_neighbors+1]  # We start from 1 to exclude self
    #     self.neighboring_indices.append(nearest_neighbors)

    def get_representations(self, trainset=True):
        """
        Extracts representations and corresponding labels from the encoder for the given dataset.

        Parameters:
            trainset (bool): If True: extracts representations from the training dataset.
                          Else testing dataset


        Updates the 'representations' and 'representations_labels'
        of shape (num_samples, representation_dim) and (num_samples,)

        """
        self.representations = []
        self.representations_labels = []
        self.encoder.eval()

        dataloader = self.train_loader if trainset else self.test_loader
        for b_idx, (data, labels) in enumerate(dataloader):
            # get image representation
            z1 = self.encoder(data.to(self.device))
            self.representations.append(z1.detach().cpu().numpy())
            self.representations_labels.append(labels.detach().cpu().numpy())
        self.representations = np.concatenate(self.representations, axis=0)
        self.representations_labels = np.concatenate(self.representations_labels, axis=0)

    # Losses:

    def Linv(self, z1, z2):
        # Invariance Loss: MSE between projections Z1 and Z2
        return F.mse_loss(z1, z2)

    def Lvar(self, z, thresh=1, eps=1e-4):
        # Variance Hinge loss on projections batch with predefined threshold
        std = torch.sqrt(z.var(dim=0) + eps)  # variance of dimensions of each projection#TODO: check dim
        hinge_loss = torch.clamp(thresh - std, min=0)  # hinge loss for each dimension
        return torch.mean(hinge_loss)  # average hinge loss across all projections (invarient to batch size)

    def Lcov(self, z):
        N, D = z.shape
        z_ = z - z.mean(dim=0)
        cov_z = ((z_.T @ z_) / (N - 1)).square()
        return (cov_z.sum() - cov_z.diagonal().sum()) / D  # TODO: is this why this loss crahsed? (added "/D") -> YES

    def loss_VICReg(self, z1, z2, lambd=25., mu=25., nu=1.):  # TODO: q4-->mu=0????
        inv_loss = self.Linv(z1, z2)
        var_loss = (self.Lvar(z1) + self.Lvar(z2))
        cov_loss = (self.Lcov(z1) + self.Lcov(z2))
        vic_loss = (inv_loss * lambd) + (var_loss * mu) + (cov_loss * nu)
        return inv_loss, var_loss, cov_loss, vic_loss

    # Linear Probing
    def linear_probing(self, num_epochs=10, num_of_classes=10, train=True):
        # function assumes self.get_representations() was called
        if train:
            # Training linear classifier
            self.encoder.eval()
            self.linear_classifier = nn.Linear(self.encoder.D, num_of_classes).to(self.device).train()

            model = nn.Sequential(self.encoder, self.linear_classifier)
            # optimize only classifier
            optimizer = torch.optim.SGD(self.linear_classifier.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()

            # Train the classifier
            for epoch in range(num_epochs):
                for i, (images, labels) in enumerate(self.train_loader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print("epoch number:" + str(epoch))
            self.save_models(encoder=False, projector=False, losses=False, linear_classifier=True)

        # Test the classifier and record accuracy
        model = nn.Sequential(self.encoder, self.linear_classifier)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        self.linear_probing_acc = accuracy
        print(f"Test Accuracy of the model on the test images: {accuracy} %")

    def plot_loss(self):
        num_batches_per_epoch_train = len(self.train_loader)
        num_batches_per_epoch_val = len(self.test_loader)
        num_epochs = len(self.train_losses['var']) // num_batches_per_epoch_train

        for loss_name in ['var', 'inv', 'cov']:
            fig = go.Figure()

            # Average train losses over each epoch
            # train_loss_averaged = [np.mean(self.train_losses[loss_name][i*num_batches_per_epoch_train:(i+1)*num_batches_per_epoch_train]) for i in range(num_epochs)]

            # Average validation losses over each epoch
            # val_loss_averaged = [np.mean(self.validation_losses[loss_name][i*num_batches_per_epoch_val:(i+1)*num_batches_per_epoch_val]) for i in range(num_epochs)]

            # Add traces
            fig.add_trace(go.Scatter(y=self.train_losses[loss_name], mode='lines', name='train'))
            # fig.add_trace(go.Scatter(y=val_loss_averaged, mode='lines', name='validation'))

            # Layout
            fig.update_layout(title=f'{loss_name} Loss over Time',
                              xaxis_title='Epochs',
                              yaxis_title='Loss')

            fig.show()

    def plot_PCA_and_SNE(self, pca_result, tsne_result):
        # Define the colorscale mapping
        color_map = ['blue', 'black', 'green', 'red', 'cyan', 'magenta', 'yellow', 'grey', 'brown',
                     'white']  # Add more if you have more classes

        # Create PCA scatter plot
        trace_pca = go.Scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            mode='markers',
            marker=dict(
                size=6,
                color=[color_map[int(i)] for i in self.representations_labels],  # assign color based on labels
                opacity=0.8),
            showlegend=False

        )

        # Create a subplot with 1 row and 2 columns
        fig = make_subplots(rows=1, cols=2)

        # Add PCA scatter plot to the first column
        fig.add_trace(trace_pca, row=1, col=1)

        # Create t-SNE scatter plot
        trace_tsne = go.Scatter(
            x=tsne_result[:, 0],
            y=tsne_result[:, 1],
            mode='markers',
            marker=dict(
                size=6,
                color=[color_map[int(i)] for i in self.representations_labels],  # assign color based on labels
                opacity=0.8),
            showlegend=False
        )

        # Add t-SNE scatter plot to the second column
        fig.add_trace(trace_tsne, row=1, col=2)

        # Update xaxis and yaxis properties for the PCA plot
        fig.update_xaxes(title_text="PCA 1", row=1, col=1)
        fig.update_yaxes(title_text="PCA 2", row=1, col=1)

        # Update xaxis and yaxis properties for the t-SNE plot
        fig.update_xaxes(title_text="t-SNE 1", row=1, col=2)
        fig.update_yaxes(title_text="t-SNE 2", row=1, col=2)

        # Now, add dummy scatter plots for the legend
        for i, color in enumerate(color_map):
            fig.add_trace(go.Scatter(
                x=[None],  # No data for x and y
                y=[None],
                mode='markers',
                marker=dict(
                    size=6,
                    color=color,
                ),
                name=f"Class {i}",  # The name that will appear in the legend
            ))

        # Update the layout
        fig.update_layout(height=600, width=1200, title_text="PCA vs t-SNE")

        # Show the plot
        fig.show()

    def retrieval_evaluation(self):

        # Extract representations
        self.get_representations()  # Also sets the encoder to evaluation mode

        # Select 10 random images from different classes
        images_from_each_class, indices_from_each_class = self.get_image_from_each_class()
        print(indices_from_each_class)

        # Get the representations of these images
        random_image_reprs = self.representations[indices_from_each_class]

        # 2. Build the FAISS index
        dimension = self.representations.shape[1]  # dimension of the vector
        index = faiss.IndexFlatL2(dimension)
        index.add(self.representations.astype('float32'))  # FAISS uses float32

        # 3. Run the nearest neighbor search with 6 nebrs

        _, indices = index.search(random_image_reprs.astype('float32'), self.representations.shape[0])
        indices = torch.tensor(indices)

        # Get the 5 nearest and furthest neighbors for each image
        for i in range(len(random_image_reprs)):
            nearest_indices = indices[i, 1:6]  # We start from 1 to exclude self
            furthest_indices = indices[i, -5:]

            nearest_images = [self.get_image_from_index(idx) for idx in nearest_indices]
            furthest_images = [self.get_image_from_index(idx) for idx in furthest_indices]

            # Plot the original image along with the nearest and furthest images
            self.plot_images([images_from_each_class[i]] + nearest_images + furthest_images)

    def get_image_from_each_class(self, train=True):
        classes = list(range(10))  # Assuming you have 10 classes
        images_from_each_class = [None] * 10
        indices_from_each_class = [None] * 10
        dataloader = self.train_loader if train else self.test_loader

        idx = 0  # Starting index
        for batch in dataloader:
            images, labels = batch
            for c in classes:
                if c in labels:
                    class_idx = (labels == c).nonzero(as_tuple=True)[0][0]
                    images_from_each_class[c] = images[class_idx].detach().cpu()
                    indices_from_each_class[c] = idx + class_idx.item()
                    classes.remove(c)
                if not classes:  # if all classes are found
                    break
            if not classes:
                break

            idx += len(images)  # Update the cumulative index

        return images_from_each_class, indices_from_each_class

    def get_image_from_index(self, idx,
                             train=True):  # TODO: refract to custom dataset with specified (__getitem__ for cleaner indexing)
        cumulative_idx = 0
        dataloader = self.train_loader if train else self.test_loader

        for batch in dataloader:
            images, labels = batch
            if cumulative_idx + len(images) > idx:
                return images[idx - cumulative_idx].cpu().detach()  # TODO: checks data stored in RAM and not GPU
            cumulative_idx += len(images)
        raise IndexError('Index out of range of dataset')

    def plot_images(self, images):
        """
        Takes a list of images and plots them in a row.
        """
        fig, axes = plt.subplots(1, len(images), figsize=(len(images), 1))

        for ax, img in zip(axes, images):
            img = img.permute(1, 2, 0)  # Transpose
            ax.imshow(img)
            ax.axis('off')

        plt.tight_layout()
        plt.show()
