import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.distributions import Normal
import numpy as np
import torch
from initialize import *
from ResnetNetworks import *
from tcn_hx import *
from torch.distributions import Categorical
from tcn_ import *
import copy 

from pretrain_CAE import ConvAutoencoder_NAV4


class NavACLNetwork(nn.Module):
    def __init__(self, task_param_dim, hidden_dim, init_w=5e-4):
        super(NavACLNetwork, self).__init__()
        self.layer_1 = nn.Linear(task_param_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_out = nn.Linear(hidden_dim, 1)

        nn.init.kaiming_uniform_(self.layer_1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer_2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer_3.weight, mode='fan_in', nonlinearity='relu')

        self.m1 = torch.nn.LeakyReLU(0.1) #torch.nn.PReLU(num_parameters=1, init=0.25)
        self.m2 = torch.nn.LeakyReLU(0.1) #torch.nn.PReLU(num_parameters=1, init=0.25)
        self.m3 = torch.nn.LeakyReLU(0.1) #torch.nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, inputs):
        x = self.m1(self.layer_1(inputs))
        x = self.m2(self.layer_2(x))
        x = self.m3(self.layer_3(x))
        x = torch.sigmoid(self.layer_out(x))

        # x = x.clamp(0,1)
    
        return (x)

    def get_task_success_prob(self, task_param_array):
        torch_task_param_array = torch.FloatTensor(task_param_array).to(device)

        difficulty_estimate = self.forward(torch_task_param_array)
    
        return difficulty_estimate.detach().cpu().numpy()

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
        
class SoftQNetwork(nn.Module):
    def __init__(self, config,):
        super(SoftQNetwork, self).__init__()
        self.config         = config
        resolutions         = config['output_resolution']
        self.num_stacked_frames = config['num_stacked_frames']

        num_lidar_beams     = config['num_lidar_beams']
        hidden_dim          = config['hidden_dim']
        num_actions         = config['num_actions']
        self._num_actions   = num_actions
        init_w              = 3e-3

        shapes              = (self.num_stacked_frames*resolutions[0],resolutions[1],resolutions[2],num_lidar_beams)
        self.inputs_shape   = (*shapes,) 

        self.kernel_size    = 2
        self.channel_s      = [256,128,64] # To think : 
        self.channel_a      = [16, 16, 16]
        self.channel_r      = [16, 16, 16]

        
        if (config['use_pretrained_vae']):
            self.config['freeze_convolution'] = True
            ae_model = ConvAutoencoder_NAV4(imgChannels=self.num_stacked_frames*resolutions[0])
            ae_model.load_state_dict(torch.load(config['pretrained_vae_path']))
            self.features=copy.deepcopy(ae_model.encode)
            del ae_model
        else:
            self.features = nn.Sequential(
                ResNetBlock(self.num_stacked_frames*resolutions[0],64,3), 
                ResNetBlock(64,128,3), 
                ResNetBlock(128,256,3), 
                ResNetBlock(256,128,3),
            )

        if(self.config['freeze_convolution']):
            # print("using fixed convolution layers for Soft Q and Policy Network !")
            for param in self.features.parameters():
                param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(self.features_size(), 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256) #  -> 64 
        )
        self.linear_Lidar_1 = nn.Linear(self.inputs_shape[3],hidden_dim) #256 -> 64
        self.linear_Lidar_2 = nn.Linear(hidden_dim,32) # 64-> 32

        self.linear_sar = nn.Linear(12,8)
        self.linear3 = nn.Linear(256 + 8 +32, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)





        
    def forward(self, state_cam, state_lidar, prev_action, prev_reward):

        state_cam =  state_cam/ 2**8
        l = self.linear_Lidar_1(state_lidar)
        l = self.linear_Lidar_2(l)
        # print(prev_reward.shape)
        # print(prev_action.shape)
        sar = torch.cat([prev_action.view(prev_action.size(0),-1), prev_reward.view(prev_reward.size(0),-1)], 1)
        # print(prev_action.view(prev_action.size(0),-1).shape)
        # print(prev_reward.view(prev_reward.size(0),-1).shape)
        # print(sar.shape)
        # print(sar.shape)
        sar = F.leaky_relu(self.linear_sar(sar))

        x = self.features(state_cam)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        # print("a, pa, rew: ", action.shape, prev_action.shape, reward.shape)
        # print("shapes x, a, pa, r, l: ", x.shape, prev_action.shape, prev_action.shape, prev_reward.shape, l.shape)

        x = torch.cat([x, sar], 1) # the dim 0 is number of samples
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        #concatenate with preprocessed lidar state
        x = torch.cat([x,l],1)
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

    def features_size(self):
        return self.features(torch.zeros(1, *self.inputs_shape[0:3])).view(1, -1).size(1)
       
        
class PolicyNetwork(nn.Module):
    def __init__(self, config):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = -20
        self.log_std_max = 2
        self.num_actions  = config['num_actions']
        self.action_range = config['action_range']
        self.config = config

        self.version     = 0
        init_w=3e-3
        self.num_stacked_frames = config['num_stacked_frames']
        resolutions         = config['output_resolution']
        shapes              = (self.num_stacked_frames*resolutions[0],resolutions[1],resolutions[2])
        num_lidar_beams     = config['num_lidar_beams']
        hidden_dim          = config['hidden_dim']
        self.inputs_shape   = (*shapes, num_lidar_beams ) 

        self.kernel_size = 2
        self.channel_s      = [256,128,64] # To think : 
        self.channel_a      = [16, 16 ,16]
        self.channel_r      = [16, 16 ,16]


        if (config['use_resnet_archi']):
                self.features = nn.Sequential(
                ResNetBlock(self.inputs_shape[0],16,3),#40*40
                ResNetBlock(16,32,3), # 18*18
                ResNetBlock(32,64,3),
                Flatten(),
                ).apply(initialize_weights_he) #7*7
        if (config['use_pretrained_vae']):
            self.config['freeze_convolution'] = True
            ae_model = ConvAutoencoder_NAV4(imgChannels=self.num_stacked_frames*resolutions[0])
            ae_model.load_state_dict(torch.load(config['pretrained_vae_path']))
            self.features=ae_model.encode
            del ae_model
        else:
            # RGB*4 x 80 x 80
            self.features = nn.Sequential(
                ResNetBlock(self.num_stacked_frames*resolutions[0],64,3), 
                ResNetBlock(64,128,3), 
                ResNetBlock(128,256,3), 
                ResNetBlock(256,128,3),
            )

        if (self.config['freeze_convolution']):
            # print("using fixed convolution layers for Policy Network !")
            for param in self.features.parameters():
                param.requires_grad = False
    
        self.fc = nn.Sequential(
            nn.Linear(self.features_size(), 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256)
        )
        self.linear_Lidar_1 = nn.Linear(self.inputs_shape[3],hidden_dim)
        self.linear_Lidar_2 = nn.Linear(hidden_dim,32)
        # L (4xSRA, 8) (8,8)
        # L (256+32+x, 2)
        # self.linear1 = nn.Linear(256 + self.num_actions +1, hidden_dim)
        # self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_sar = nn.Linear(12,8)
        self.linear3 = nn.Linear(256 + 8 +32, hidden_dim)
        # self.linear4 = nn.Linear(hidden_dim, hidden_dim)


        self.mean_linear = nn.Linear(hidden_dim, self.num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_dim, self.num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)


        
    def forward(self, state_cam, state_lidar,prev_action, prev_reward):
        state_cam   = state_cam / 2**8
        state_lidar = state_lidar
        l = self.linear_Lidar_1(state_lidar)
        l = self.linear_Lidar_2(l)
        
        sar = torch.cat([prev_action.view(prev_action.size(0),-1), prev_reward.view(prev_reward.size(0),-1)], 1)
        # print(prev_action.view(prev_action.size(0),-1).shape)
        # print(prev_reward.view(prev_reward.size(0),-1).shape)
        # print(sar.shape)
        sar = F.leaky_relu(self.linear_sar(sar))

        x = self.features(state_cam)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        # print("shapes x, a, pa, r: ", x.shape, prev_action.shape, reward.shape)
        x = torch.cat([x, sar], 1) # the dim 0 is number of samples
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))

        # concatenate with preprocessed lidar state
        x = torch.cat([x,l],1)
        x = F.relu(self.linear3(x))
        # x = F.relu(self.linear4(x))

        mean    = (self.mean_linear(x))
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state_cam, state_lidar,prev_action, prev_reward, device, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std = self.forward(state_cam, state_lidar,prev_action, prev_reward)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z      = normal.sample(mean.shape) 
        action_0 = torch.tanh(mean + std*z.to(device)) # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range*action_0
        # The log-likelihood here is for the TanhNorm distribution instead of only Gaussian distribution. \
        # The TanhNorm forces the Gaussian with infinite action range to be finite. \
        # For the three terms in this log-likelihood estimation: \
        # (1). the first term is the log probability of action as in common \
        # stochastic Gaussian action policy (without Tanh); \
        # (2). the second term is the caused by the Tanh(), \
        # as shown in appendix C. Enforcing Action Bounds of https://arxiv.org/pdf/1801.01290.pdf, \
        # the epsilon is for preventing the negative cases in log; \
        # (3). the third term is caused by the action range I used in this code is not (-1, 1) but with \
        # an arbitrary action range, which is slightly different from original paper.
        log_prob = Normal(mean, std).log_prob(mean+ std*z.to(device)) - torch.log(1. - action_0.pow(2) + epsilon) -  np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action); 
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability, 
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state_cam, state_lidar,prev_action, prev_reward,  deterministic, device):

        state_cam   = torch.FloatTensor(state_cam).unsqueeze(0).to(device) 
        state_lidar = torch.FloatTensor(state_lidar).unsqueeze(0).to(device) 
        prev_action = torch.FloatTensor(prev_action).unsqueeze(0).to(device) # needs to fit state dimensions so we add 1 dim
        prev_reward      = torch.FloatTensor(prev_reward).unsqueeze(0).to(device) # needs to fit state dimensions so we add 1 dim


        mean, log_std = self.forward(state_cam, state_lidar,prev_action, prev_reward)
        std = log_std.exp()
        

        normal = Normal(0, 1)
        z      = normal.sample(mean.shape).to(device)
        action = self.action_range* torch.tanh(mean + std*z)
        

        action = self.action_range* torch.tanh(mean).detach().cpu().numpy()[0] if deterministic else action.detach().cpu().numpy()[0]
        # print(action)
        return action


    def sample_action(self,):
        a=torch.FloatTensor(self.num_actions).uniform_(-1, 1)
        return self.action_range*a.numpy()
    def features_size(self):
        return self.features(torch.zeros(1, *self.inputs_shape[0:3])).view(1, -1).size(1)


# -*- coding: utf-8 -*-
"""
Created on Tue May 25 17:41:51 2021

@author: Honghu Xue
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import initialize_weights_he
from utils import initialize_weights_xavier
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)    
    
    
class ResNetBlock(nn.Module):
    '''https://www.reddit.com/r/MachineLearning/comments/671455/d_batch_normalization_in_reinforcement_learning/
        Batch normalziation causes instability in training in DQN
        Weight normalziation can increase a little bit on the performance in DQN. https://proceedings.neurips.cc/paper/2016/file/ed265bc903a5a097f61d3ec064d96d2e-Paper.pdf
    '''
    
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResNetBlock, self).__init__()
        # !!! Don't fully understand what is padding = same in the slide!!!
        self.residual_pass = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                                    stride = 1, padding = 1, padding_mode='replicate'),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                                    stride = 1, padding = 1, padding_mode='replicate'), 
                               )
        self.identity_mapping = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                                    stride = 1), # no padding here
                               )
        # After output size after passing through identity_mapping is W_2 = (signal_length - Kernal_size +2 * Padding)/Stride + 1        
        self.down_dimension = nn.Sequential(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=2,
                                                    stride = 2, padding = 0),
                               )                  
        self.RelU = nn.ReLU()
        
    def forward(self, x):
        #TODO
        print("orig: ", x.shape)
        residual = self.residual_pass(x)
        print("residual: ", residual.shape)
        x = self.identity_mapping(x) 
        print("identiy: ", x.shape)
        x += residual
        print("union: ", x.shape)
        x =  self.RelU(x)
        x = self.down_dimension(x)
        print("downdim: ", x.shape)

        return x