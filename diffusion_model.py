######################################################## IMPORTS ######################################################
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import plotly.express as px

# Setting reproducibility
# ensuring that all the random operations performed by these modules are deterministic


SEED = 379
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
TRAINING_SAMPLES_COUNT = 3000  # increaet this form 3000

SAMPLES_DIM = 2
TIME_EMBEDDING_SIZE = 1
DENOISER_TRAINING_LEARNING_RATE = 0.007
DENOISER_TRAINING_EPOCHS = 1000
DENOISER_TRAINING_BATCH_SIZE = 128


# time_steps->T
# scheduler->σ

################################################# FUNCTIONS ###########################################################
# 2.1.1 The Forward Process #TODO: do we need to write code for forward procerss?

def sample_datapoints(a: int, b: int, samples_num: int, samples_dim: int):
    '''
    Function to generate points uniformly,
    inside a square, ranging from x = [−1,1], y = [−1,1]
    :param samples_num:
    :return: np.tensor
    '''
    return a + (b - a) * torch.rand(samples_num, samples_dim)


def exp_scheduler(t):
    '''
    Noise schedule function
    :param t: timestep
    :return: standard deviation of the noise added at each timestep
    '''
    return torch.exp(5 * (t - 1))
def sigmoid_scheduler(t):
    '''
    Sigmoid noise schedule function
    This scheduler increases slowly at first, more quickly in the middle, and then slows again towards the end.
    This could be helpful when your model needs to adapt to a variety of noise levels throughout training
    :param t: timestep
    :return: standard deviation of the noise added at each timestep
    '''
    return torch.sigmoid(t)
def sqrt_scheduler(t):
    '''
    Square root noise schedule function
    These schedulers can be useful if you want the amount of noise added to increase slowly.
    This might be appropriate for problems where adding too much noise early in the training process could be detrimental.
    :param t: timestep
    :return: standard deviation of the noise added at each timestep
    '''
    return torch.sqrt(t)
def derived_scheduler(t):
    '''
    Noise schedule function derived according to t
    :param t: timestep
    :return: standard deviation of the noise added at each timestep
    '''
    return 5 * torch.exp(5 * (t - 1))



class Denoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # #flattern
        self.linear_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 8),
            nn.LeakyReLU(),
            nn.Linear(8, output_dim))

    def forward(self, x, time_step):
        time_step = time_step.float()  # convert long -> float before concatenating
        x = torch.cat([x, time_step], dim=-1)  # Condition on timestep by concatenating
        x = self.linear_layer(x)
        return x


class DiffusionDenoiser(nn.Module):
    def __init__(self, time_steps=1000):
        super(DiffusionDenoiser, self).__init__()

        self.time_step = time_steps
        self.denoiser = Denoiser(input_dim=TIME_EMBEDDING_SIZE + SAMPLES_DIM, hidden_dim= 16,
                                 output_dim=SAMPLES_DIM)
        self.embedding = nn.Embedding(num_embeddings=time_steps,
                                      embedding_dim=TIME_EMBEDDING_SIZE)  # one time embedding for all sample

    def forward(self, x, t):
        # te = self.embedding(t.long())
        return self.denoiser(x, t)  # Pass through denoiser


    def save_model(self, path='./denoiser_model/denoiser.pt'):
        torch.save(self.denoiser, path)

    def train_denoiser(self,dataloader,optimizer,scheduler, loss_function=nn.MSELoss(),
                       epochs=DENOISER_TRAINING_EPOCHS, batch_size=DENOISER_TRAINING_BATCH_SIZE):
        '''
        The Reverse Process - Training Denoiser
        return: losses for visualization
        '''
        self.denoiser.train()  # switch the model to training mode
        losses = []
        for epoch in range(epochs):
            for datapoints in dataloader:
                # forward process:
                # 1. Sample Gaussian noise ε ∼ N (0, I)
                noise = torch.randn(*datapoints.shape)
                # 2. Sample a random time t ∈ [0, T] T=timestep
                t = torch.rand(size=(datapoints.shape[0], 1))
                # 3. Obtain xt using xt=x0+σ2(t)·ε ε∼N(0,I)
                xt = datapoints + scheduler(t) * noise  # noisy points xt
                # 4. Backpropagate objective
                optimizer.zero_grad()
                prediction = self.denoiser(xt, t)
                loss = loss_function(prediction, noise)

                loss.backward()
                optimizer.step()
                losses.append(loss.item())

        return losses

    def evaluate_denoiser(self,dataloader, scheduler, loss_function=nn.MSELoss()):
        self.denoiser.eval()  # switch the model to evaluation mode
        validation_losses = []
        with torch.no_grad():  # disable gradients computation
            for datapoints in dataloader:
                # 1. Sample Gaussian noise ε ∼ N (0, I)
                noise = torch.randn(*datapoints.shape)

                # 2. Sample a random time t ∈ [0, T] T=timestep
                t = torch.rand(size=(datapoints.shape[0], 1))

                # 3. Obtain xt using Eq. 4
                xt = datapoints + scheduler(t) * noise

                # Compute objective: l2(D(xt,t),ε)
                prediction = self.denoiser(xt, t)
                loss = loss_function(prediction, noise)

                validation_losses.append(loss.item())

        return validation_losses # return the average loss


#######################################################################################################################
# NOTE: no batch sice we process all datapoints


# 2.1.3 The Reverse Process - DDIM Sampling

# TODO: we have varience exploding so we need to sample not from z ∼ N (0, I)???
def DDIM_sampling(denoiser: DiffusionDenoiser, samples_num: int = TRAINING_SAMPLES_COUNT,
                  samples_dim: int = SAMPLES_DIM):
    '''
    The Reverse Process - DDIM Sampling
    :param denoiser: denoiser neural network
    :param samples_num: number of samples
    :param samples_dim: dimension of each sample (2) for 2D-datapoints
    :return: predicted sample vector z
    '''
    # Sample z ∼ N (0, I)
    z = torch.randn((samples_num, samples_dim))

    # Iterate from t = 1 to 0 with step dt
    dt = 1 / denoiser.time_step
    for t in np.arange(1, 0, -dt):
        # we are interested in how the distribution changes over time
        # thus we will pass the same t to all our samples
        t = torch.tensor(t).repeat(samples_num, 1)

        # Estimate noise:
        t_tensor = torch.tensor([int(t * 1000)], dtype=torch.long)
        time_embedding = denoiser.embedding(t_tensor)
        estimated_noise = denoiser(z, time_embedding)

        # Estimate denoise x_hat
        x_hat = z - exp_scheduler(t) * estimated_noise

        # Calculate the score function
        score_z = (x_hat - z) / (exp_scheduler(t) ** 2)
        # Update z with the reverse process
        dz = derived_scheduler(t) * exp_scheduler(t) * score_z * dt  # closed formula for reverse process
        z = z + dz

    return z


def estimate_probability(x, denoiser, scheduler, num_samples=1000):
    '''
    The Reverse Process - Probability Estimation
    :param x: point to estimate
    :param denoiser: denoiser neural network
    :param scheduler: noise scheduler function
    :param num_samples: qty of samples used for estimation
    :return: ELBO log(p(x))
    '''
    # Number of time steps
    T = denoiser.time_step

    # 1. Randomize many possible noise and time combinations
    noise = torch.randn(num_samples, *x.shape)
    t = torch.rand(num_samples, *x.shape)  # x,shape = [2,1] #TODO: check if this is right

    # 2. Perform a forward process for all the combinations starting from x as the input
    xt = x + scheduler(t) * noise

    # 3. Estimate x0 for all combinations
    x0_hat = denoiser(xt, t)

    # 4. Compute SNR differences for the sampled t values
    dt = torch.ones_like(t) / T  # Assume constant dt for simplicity
    SNR_diff = scheduler(t - dt.abs()) - scheduler(t)

    # Compute the squared L2 distance ||x - x0_hat||²
    l2_distance = torch.sum((x - x0_hat) ** 2, dim=-1)

    # Compute the expectation over the noise and time samples
    expectation = torch.mean(SNR_diff * l2_distance)

    # 5. Average the results and multiply by T
    LT = (T * expectation) / 2

    # 6. -LT(x) is the lower bound for log p(x)
    log_p_x = -LT

    return log_p_x


def sample_point_and_plot_trajectory(time_steps=1000):
    '''
    sampling a point and performing forward process.
    visualizing the trajectory in 2D space.
    :return: None
    '''
    successive_points = []
    initial_point = sample_datapoints(-1, 1, 1, 2)
    for t in np.arange(0, 1, 1 / time_steps):
        t = torch.tensor(t).repeat(1, 1)
        xt = initial_point + exp_scheduler(t) * torch.randn(1, 2)  # (xt=x0+σ2(t)·ε ε∼N(0,I))
        successive_points.append(xt)
    successive_points = torch.stack(successive_points).squeeze()
    df = pd.DataFrame(successive_points.numpy(), columns=['x', 'y'])
    df['timestep'] = range(len(df))

    # trajectory in a 2D space
    fig = px.scatter(df, x='x', y='y', color='timestep',
                     opacity=np.linspace(0.1, 1, len(df)),
                     color_continuous_scale=px.colors.sequential.Jet,
                     labels={'color': 'Time step'}, title="Forward process of a point as a trajectory in 2D space")
    fig.show()


if __name__ == '__main__':
    # q1
    # sample_point_and_plot_trajectory(1000)
    # Q2
    training_samples = sample_datapoints(a=-1, b=1, samples_num=TRAINING_SAMPLES_COUNT, samples_dim=SAMPLES_DIM)
    dataloader = DataLoader(training_samples, batch_size=DENOISER_TRAINING_BATCH_SIZE, shuffle=True)
    denoiser = DiffusionDenoiser(time_steps=1000)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=DENOISER_TRAINING_LEARNING_RATE)
    train_losses = denoiser.train_denoiser(dataloader,optimizer,exp_scheduler)
    validation_losses = denoiser.evaluate_denoiser(dataloader,exp_scheduler)

    denoiser.save_model(path='denoiser.pt')

    df1 = pd.DataFrame(data={'training_step': range(1, len(train_losses) + 1), 'loss': train_losses})
    fig1 = px.line(df1, x='training_step', y='loss',
                  title='loss function over the training batches of the denoiser')
    fig1.show()

    df2 = pd.DataFrame(data={'validation_step': range(1, len(validation_losses) + 1), 'loss': validation_losses})
    fig2 = px.line(df2, x='validation_step', y='loss',
                  title='loss function over the evaluation batches  of the denoiser')
    fig2.show()
    print(sum(validation_losses)/len(validation_losses))
#     #Q3
#     # Create subplot grid
#     fig = make_subplots(rows=3, cols=3)
#
#     # seed values
#     seeds = np.arange(1, 10)
#
#     # For each subplot, set a different seed and generate a scatter plot
#     for i in range(3):
#         for j in range(3):
#             np.random.seed(seeds[i * 3 + j])  # set the seed
#             x = np.random.rand(1000)  # generate 1000 random x values
#             y = np.random.rand(1000)  # generate 1000 random y values
#
#             # Create scatter plot
#             scatter = go.Scatter(
#                 x=x,
#                 y=y,
#                 mode="markers",
#                 name=f'Seed: {seeds[i * 3 + j]}'
#             )
#
#             # Add scatter plot to the correct subplot
#             fig.add_trace(scatter, row=i + 1, col=j + 1)
#
#     fig.show()
#
# # modeling a :
#
# class Denoiser(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         # #flattern
#         self.linear_layer = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LeakyReLU(),
#             nn.Linear(hidden_dim, 8),
#             nn.LeakyReLU(),
#             nn.Linear(8, output_dim),
#         )