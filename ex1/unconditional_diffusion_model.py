import torch

from diffusion_model import *
from helpers_and_metrics import *

MODEL_PATH = "/denoiser_dropout.pt"


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


def generate_samples_and_grid(denoiser, scheduler, samples_num: int, samples_dim: int):
    '''
    generate samples from the denoiser and plot them in a grid
    :param denoiser: denoiser neural network
    :param scheduler: noise scheduler function
    :param samples_num: qty of samples to generate
    :param samples_dim: dimension of each sample
    :return: None
    '''

    # Create subplot grid
    fig = make_subplots(rows=3, cols=3)

    for i in range(1, 10):
        # For each subplot, set a different seed and generate a scatter plot
        torch.manual_seed(i)
        z = denoiser.DDIM_sampling(exp_scheduler, samples_num=128, samples_dim=2)
        # Create scatter plot and add to figure
        fig.add_trace(
            go.Scatter(
                x=z[:, 0].detach().numpy(),
                y=z[:, 1].detach().numpy(),
                mode="markers",
                name=f'Seed: {i}'
            ),
            row=(i - 1) // 3 + 1,
            col=(i - 1) % 3 + 1
        )

    fig.show()


if __name__ == '__main__':
    # Q1
    # sample_point_and_plot_trajectory(1000)
    # Q2
    # training_samples = sample_datapoints(a=-1, b=1, samples_num=TRAINING_SAMPLES_COUNT, samples_dim=SAMPLES_DIM)
    # dataloader = DataLoader(training_samples, batch_size=DENOISER_TRAINING_BATCH_SIZE, shuffle=True)
    # denoiser = DiffusionDenoiser(time_steps=1000)
    # optimizer = torch.optim.Adam(denoiser.parameters(), lr=DENOISER_TRAINING_LEARNING_RATE)
    # train_losses = denoiser.train_denoiser(dataloader,optimizer,)
    # denoiser.save_model(path='denoiser.pt')

    # df1 = pd.DataFrame(data={'training_step': range(1, len(train_losses) + 1), 'loss': train_losses})
    # fig1 = px.line(df1, x='training_step', y='loss',
    #               title='loss function over the training batches of the denoiser')
    # fig1.show()

    # Q3
    # denoiser = DiffusionDenoiser(time_steps=1000,denoiser_path="/Users/matancohen/Desktop/AML/denoiser_model/denoiser.pt")
    # generate_samples_and_grid(denoiser,exp_scheduler,samples_num=128,samples_dim=2)

    # Q4
    # fixed_point = torch.tensor([0, 0]).repeat(128,1)
    # time_steps_variations = [0,100,200,300,400,500,600,700,800,900,1000,10000] #original point when t=0
    # resulting_points = [fixed_point[0]]
    # for time_steps in time_steps_variations[1:]:
    #     denoiser = DiffusionDenoiser(time_steps=time_steps,
    #                                  denoiser_path=MODEL_PATH)
    #     new_points = denoiser.DDIM_sampling(exp_scheduler, samples_num=128, samples_dim=2, fixed_point=fixed_point)
    #     resulting_points.append(new_points[0])
    #
    # # creating dataframe  to convert time_ses_variations into categorical type
    # df = pd.DataFrame({
    #     'x': [point[0].item() for point in resulting_points],
    #     'y': [point[1].item() for point in resulting_points],
    #     'Timesteps': pd.Categorical(time_steps_variations)  # converting time_steps_variations to categorical type
    # })
    #
    # fig = px.scatter(df,
    #                  x='x',
    #                  y='y',
    #                  color='Timesteps',
    #                  labels={'x': 'x', 'y': 'y', 'Timesteps': 'Timesteps'},
    #                  title="DDIM sampling of fixed point with different Ts")
    #
    # fig.show()

    # Q5
    # T = 1000 # total time steps
    # dt = 1/T # time step
    # t = torch.tensor(np.arange(0, 1, dt))
    # exp_s = exp_scheduler(t)
    # sig_s = sigmoid_scheduler(t)
    # sqr_s = sqrt_scheduler(t)
    # derived_exp_s = derived_scheduler(t)
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=t, y=exp_s, mode= "lines",name='exp_scheduler'))
    # fig.add_trace(go.Scatter(x=t, y=sig_s, mode= "lines",name='sigmoid_scheduler'))
    # fig.add_trace(go.Scatter(x=t, y=sqr_s, mode= "lines",name='sqrt_scheduler'))
    # fig.add_trace(go.Scatter(x=t, y=derived_exp_s, mode= "lines",name='derived_exp_scheduler'))
    # fig.show()

    # Q6
    # Perform reverse sampling process multiple times with the same input noise
    # initial_noise = sample_datapoints(a=1,b=-1,samples_num=1,samples_dim=2)  # Same input noise for all runs
    # samples = []
    #
    # for i in range(10):
    #     denoiser = DiffusionDenoiser(time_steps=1000,denoiser_path=MODEL_PATH)
    #     sample = denoiser.DDIM_sampling(exp_scheduler, samples_num=1, samples_dim=2, fixed_point=initial_noise)
    #     samples.append(sample.detach().numpy())

    # fig = go.Figure()
    # for i, sample in enumerate(samples):
    #     fig.add_trace(go.Scatter(x=sample[:, 0], y=sample[:, 1], mode='markers', name=f'Run {i + 1}'))
    #
    # fig.update_layout(title='Outputs of reverse sampling process with same input noise',
    #                   xaxis_title='X',
    #                   yaxis_title='Y',
    #                   legend_title='Run')
    #
    # fig.show()

    # denoise the trajectories of four points
    # Select four points to denoise
    # num_points_to_denoise = 1000
    #
    # # Denoise the trajectories and plot the results
    # fig = go.Figure()
    #
    # for i in range(num_points_to_denoise):
    #     denoiser = DiffusionDenoiser(time_steps=1000)
    #     trajectory = denoiser.DDIM_sampling(exp_scheduler, samples_num=1, samples_dim=2,
    #                                         fixed_point=initial_noise,trajectory=[initial_noise])
    #     fig.add_trace(go.Scatter(x=[point[0][0] for point in trajectory], y=[point[0][1] for point in trajectory],
    #                              mode='lines', name=f'Trajectory {i + 1}'))
    # fig.update_layout(title='Denoising trajectories of four points',
    #                   xaxis_title='X',
    #                   yaxis_title='Y',
    #                   legend_title='Trajectory')
    # fig.show()

    ## Conditional Diffusion Denoiser
    # Training
    # torch.manual_seed(3)
    training_samples = sample_datapoints(a=-1, b=1, samples_num=TRAINING_SAMPLES_COUNT, samples_dim=SAMPLES_DIM)
    cond_data = Conditional_dataset(training_samples)
    # print(cond_data.data_distribution)
    #
    # dataloader = DataLoader(cond_data, batch_size=DENOISER_TRAINING_BATCH_SIZE, shuffle=True)
    # cond_denoiser = Cond_DiffusionDenoiser(time_steps=1000,
    #                                        time_embedding_size=16,
    #                                        samples_dim=SAMPLES_DIM,
    #                                        num_classes= len(cond_data.class_dict),
    #                                        class_embedding_size=16,
    #                                        denoiser_path=None)
    # optimizer = torch.optim.Adam(cond_denoiser.parameters(), lr=DENOISER_TRAINING_LEARNING_RATE)
    # train_losses = cond_denoiser.train_denoiser(dataloader,optimizer,exp_scheduler)
    # print(np.mean(train_losses))
    # cond_denoiser.save_model(path='denoiser_conditional.pt')
    #
    # df1 = pd.DataFrame(data={'training_step': range(1, len(train_losses) + 1), 'loss': train_losses})
    # fig1 = px.line(df1,
    #                x='training_step',
    #                y='loss',
    #                title='loss function over the training batches of the denoiser')
    # fig1.show()

    # Q1
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'black', 'cyan', 'magenta']
    fig = go.Figure()
    for class_idx in range(10):
        class_points = cond_data.get_class_points(class_idx)
        fig.add_trace(
            go.Scatter(
                x=class_points[:, 0],
                y=class_points[:, 1],
                mode='markers',
                name=f'Class {class_idx}',
                marker=dict(size=5, color=colors[class_idx]
                            )
            ))
    fig.show()

    # Q2
    # get sample for each class
    # cond_denoiser = Cond_DiffusionDenoiser(time_steps=1000,
    #                               time_embedding_size=16,
    #                               samples_dim=SAMPLES_DIM,
    #                               num_classes=len(cond_data.class_dict),
    #                               class_embedding_size=16,
    #                               denoiser_path='denoiser_conditional.pt')
    # fig = go.Figure()
    # for class_idx in cond_data.class_dict.keys():
    #     sample = cond_data.get_class_points(class_idx)[0]
    #     sample = sample[None,:]
    #     trajectory = cond_denoiser.DDIM_sampling(exp_scheduler,
    #                                         samples_num=1,
    #                                         samples_dim=2,
    #                                         sample_points=sample,
    #                                         sample_classes = torch.tensor([class_idx]),
    #                                         trajectory=[sample])
    #     fig.add_trace(go.Scatter(x=[point[0][0] for point in trajectory], y=[point[0][1] for point in trajectory],
    #                              mode='lines', name=f'Trajectory {class_idx + 1}'))
    # fig.update_layout(title='Denoising trajectories of points in classes',
    #                   xaxis_title='X',
    #                   yaxis_title='Y',
    #                   legend_title='Trajectory')
    # fig.show()

    # Q4
    fig = go.Figure()
    training_samples = sample_datapoints(a=-1, b=1, samples_num=1000, samples_dim=SAMPLES_DIM)
    cond_data = Conditional_dataset(training_samples)
    print(cond_data.data_distribution)
    cond_denoiser = Cond_DiffusionDenoiser(time_steps=1000,
                                           time_embedding_size=16,
                                           samples_dim=SAMPLES_DIM,
                                           num_classes=len(cond_data.class_dict),
                                           class_embedding_size=16,
                                           denoiser_path='denoiser_model/denoiser_conditional.pt')
    z = cond_denoiser.DDIM_sampling(exp_scheduler,
                                    samples_num=1000,
                                    samples_dim=2,
                                    sample_points=cond_data.samples,
                                    sample_classes=cond_data.labels,
                                    trajectory=None)
    # Create scatter plot and add to figure
    colors = np.array(['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'black', 'cyan', 'magenta'])
    colors_array = colors[cond_data.labels.numpy()]
    fig.add_trace(
        go.Scatter(
            x=z[:, 0].detach().numpy(),
            y=z[:, 1].detach().numpy(),
            mode="markers",
            marker=dict(
                size=6,
                color=colors_array,  # set color equal to a variable
            ),
        ),
    )

    fig.show()
    # Q6
