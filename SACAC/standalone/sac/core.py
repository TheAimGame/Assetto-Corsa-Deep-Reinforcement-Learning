import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
def combined_shape(length, shape=None):
    """
    Helper function to combine a batch size with a given shape.
    Returns (length,) if shape is None, or (length, shape...) otherwise.
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)
def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Constructs a multi-layer perceptron using the specified sizes and activation functions.
    Args:
        sizes: list of layer sizes including input and output
        activation: activation function for hidden layers
        output_activation: activation function for the output layer (default is Identity)
    Returns:
        A torch.nn.Sequential MLP model
    """
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
def count_vars(module):
    """
    Counts the number of trainable parameters in a PyTorch module.
    """
    return sum([np.prod(p.shape) for p in module.parameters()])

LOG_STD_MAX = 2
LOG_STD_MIN = -20
class SquashedGaussianMLPActor(nn.Module):
    """
    The policy (actor) network that outputs squashed Gaussian-distributed actions.
    A Gaussian-distributed action means that the actions your reinforcement learning agent chooses come from a normal (bell curve) probability distribution, rather than being picked directly or deterministically
    encourages exploration.
    The squashing function (tanh) is used to ensure that the actions are within the action space limits.
    The actor network is a multi-layer perceptron (MLP) that takes the observation as input and outputs the mean and log standard deviation of the Gaussian distribution.
    obs_dim: Size of observation vector.

act_dim: Size of action vector.

hidden_sizes: Neural network structure.

act_limit: Maximum absolute action value (from action_space.high).
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        
        self.act_limit = act_limit
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    def forward(self, obs, deterministic=False, with_logprob=True):
        """
        Given observations, returns an action and optionally the log probability of that action.
        Args:
            obs: observation input
            deterministic: whether to use deterministic (mean) actions
            with_logprob: whether to return log probability
        Returns:
            action, log_prob (if with_logprob=True)
        """
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        
        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu  
        else:
            pi_action = pi_distribution.rsample()  
        if with_logprob:
            
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None
        
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        return pi_action, logp_pi
class MLPQFunction(nn.Module):
    """
    Q-function (critic) that estimates Q-values for (obs, act) pairs.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation) #Builds a simple MLP that takes [obs, act] as a concatenated input vector, and outputs a single scalar value.
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    def forward(self, obs, act):
        """
        Estimate Q-value for the given observation and action.
        Args:
            obs: observation
            act: action
        Returns:
            Q-value as a scalar
        """
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  
class MLPActorCritic(nn.Module):
    """
    Combined Actor-Critic module. Includes:
        - A squashed Gaussian policy (actor)
        - Two Q-networks (critics) for double Q-learning
    """
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256),
                 activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        
        self.pi = SquashedGaussianMLPActor(
            obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    def act(self, obs, deterministic=False):
        """
        Get action from the policy without computing gradients.
        Args:
            obs: observation (PyTorch tensor)
            deterministic: whether to use deterministic actions
        Returns:
            Numpy action array
        used for inference - choosing actions based on the policy without updating the model.
        """
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.numpy()
