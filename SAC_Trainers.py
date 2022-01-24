from hashlib import new
from torch import ne
from Networks import *
from per_buffer import *
from utils    import *
from ResnetNetworks import *



class SAC_Trainer():
    def __init__(self,save_location, config):
        self.save_config             = config
        self.load_config(self.save_config)
        save_location           = save_location
        self.q1_losses          = []
        self.q2_losses          = []
        self.policy_losses      = []
        self.alpha_losses       = []
        self.alpha_list         = []
        self.update_step        = 0
        self.steps              = 0
        action_range            = 1.


        input_shape             = (self.input_shape[0],self.input_shape[1],self.input_shape[2])
        action_dim              = 2
        state_dim               = (self.num_stacked_frames*input_shape[0],input_shape[1],input_shape[2],self.num_lidar_beams)

        self.soft_q_net1            = SoftQNetwork(config).to(train_device)
        self.soft_q_net2            = SoftQNetwork(config).to(train_device)
        self.target_soft_q_net1     = SoftQNetwork(config).to(train_device)
        self.target_soft_q_net2     = SoftQNetwork(config).to(train_device)
        self.policy_net             = PolicyNetwork(config).to(train_device)
        self.log_alpha              = torch.ones(1, device=train_device) * np.log(self.alpha)
        self.log_alpha.requires_grad= True



        if(self.use_exp_prio_buffer):
            self.replay_buffer     = PrioritizedReplayBuffer(self.buffer_maxlen,config,mem_efficient_mode=True,state_dim=state_dim,action_dim=2,num_stacked_frames=self.num_stacked_frames,alpha=config['prioritized_replay_alpha']) 
            self.soft_q_criterion1 = torch.nn.SmoothL1Loss(reduction = 'none')
            self.soft_q_criterion2 = torch.nn.SmoothL1Loss(reduction = 'none')
            self.beta_schedule     = LinearSchedule(config['buffer_maxlen'],
                                                    initial_p=config['prioritized_replay_beta0'],
                                                    final_p=config['prioritized_replay_beta1'])

        else:
            self.replay_buffer     = ReplayBuffer(self.buffer_maxlen,config,mem_efficient_mode=True,state_dim=state_dim,action_dim=2,num_stacked_frames=self.num_stacked_frames) #(self.buffer_maxlen,self.num_stacked_frames,state_dim) # 
            self.soft_q_criterion1 = torch.nn.SmoothL1Loss(reduction = 'none')
            self.soft_q_criterion2 = torch.nn.SmoothL1Loss(reduction = 'none')



        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)


        # for freezing feature extractor, dont put in to optimizer
        self.soft_q_optimizer1 = optim.Adam(filter(lambda p: p.requires_grad, self.soft_q_net1.parameters()), lr=self.q_lr)
        self.soft_q_optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, self.soft_q_net2.parameters()), lr=self.q_lr)
        self.policy_optimizer  = optim.Adam(filter(lambda p: p.requires_grad, self.policy_net.parameters()),  lr=self.p_lr)

        self.alpha_optimizer   = optim.Adam([self.log_alpha], lr=self.a_lr)


    def all_elements_to(self, target_device):
        self.soft_q_net1            = self.soft_q_net1.to(target_device)
        self.soft_q_net2            = self.soft_q_net2.to(target_device)
        self.target_soft_q_net1     = self.target_soft_q_net1.to(target_device)
        self.target_soft_q_net2     = self.target_soft_q_net2.to(target_device)
        self.policy_net             = self.policy_net.to(target_device)
        self.log_alpha              = self.log_alpha.to(target_device)

    def load_config(self,config):
        self.tau                     = config['tau']
        self.alpha                   = config['alpha']
        self.q_lr                    = config['q_lr']
        self.p_lr                    = config['p_lr']
        self.a_lr                    = config['a_lr']
        self.buffer_maxlen           = config['buffer_maxlen']
        self.num_stacked_frames      = config['num_stacked_frames']
        self.hidden_dim              = config['hidden_dim']
        self.freeze_convolution      = config['freeze_convolution']
        self.use_grad_clip           = config['use_grad_clip']
        self.max_grad                = config['max_grad']
        self.use_exp_prio_buffer     = config['use_exp_prio_buffer']
        self.use_hard_update         = config['use_hard_update']
        self.use_pretrained_resnet18 = config['use_pretrained_resnet18']
        self.num_lidar_beams         = config['num_lidar_beams']
        self.input_shape             = config['output_resolution']
        self.hard_update_n_step      = config['hard_update_n_step']
    
    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-1, gamma=0.99,soft_tau=1e-3, use_reward_scaling=False):
        start = current_milli_time()
        if self.use_exp_prio_buffer == False:
            c_obs,l_obs, action, prev_action, reward, prev_reward, c_next_osb,l_next_obs, done = self.replay_buffer.sample(batch_size)
        else:
            c_obs,l_obs, action, prev_action, reward, prev_reward, c_next_osb,l_next_obs, done, weights, idxs = self.replay_buffer.sample(batch_size,self.beta_schedule.value(self.update_step))
            weights = torch.FloatTensor(weights).to(train_device)

        # print("sample time: ", current_milli_time()- start)

        self.update_step +=1 
        cam_tensor          = torch.FloatTensor(c_obs).to(train_device)
        lidar_tensor        = torch.FloatTensor(l_obs).to(train_device)
        next_cam_tensor     = torch.FloatTensor(c_next_osb).to(train_device)
        next_lidar_tensor   = torch.FloatTensor(l_next_obs).to(train_device)
        action              = torch.FloatTensor(action).to(train_device)
        prev_action         = torch.FloatTensor(prev_action).to(train_device)
        reward              = torch.FloatTensor(reward).to(train_device)       # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim; - no now it is a stack of 4 so we dont need unsqzeeze anymore!
        prev_reward         = torch.FloatTensor(prev_reward).to(train_device)  # prev_reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim; - no now it is a stack of 4 so we dont need unsqzeeze anymore!
        done                = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(train_device)

        predicted_q_value1 = self.soft_q_net1(cam_tensor,lidar_tensor, prev_action, prev_reward)
        predicted_q_value2 = self.soft_q_net2(cam_tensor,lidar_tensor, prev_action, prev_reward)

        with torch.no_grad():
            new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_cam_tensor,next_lidar_tensor,action, reward, device=train_device) 
            new_next_stacked_action = action.view(action.size(0),-1)[:,2:]
            new_next_stacked_action = torch.cat((new_next_stacked_action,new_next_action),dim=1)
            target_q_min = torch.min(self.target_soft_q_net1(next_cam_tensor,next_lidar_tensor, new_next_stacked_action, reward),self.target_soft_q_net2(next_cam_tensor, next_lidar_tensor, new_next_stacked_action,reward)) - self.alpha * next_log_prob
            target_q_value = reward[:,-1] + (1 - done) * gamma * target_q_min # if done==1, only reward

        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(cam_tensor,lidar_tensor,prev_action, prev_reward, device=train_device)
        new_stacked_action = prev_action.view(prev_action.size(0),-1)[:,2:]
        new_stacked_action = torch.cat((new_stacked_action,new_action),dim=1)
        new_action         = new_stacked_action
        if(use_reward_scaling):
            reward      = reward[:,-1]
            prev_reward = prev_reward[:,-1]

            prev_reward = reward_scale * (prev_reward - prev_reward.mean(dim=0)) / (prev_reward.std(dim=0) + 1e-6)
            reward      = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

    # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
            self.alpha_losses.append(alpha_loss.item())
            self.alpha_list.append(self.alpha.item())

        else:
            self.alpha = self.alpha
            alpha_loss = 0

        if self.use_exp_prio_buffer == False:            
            q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach()).squeeze()  # detach: no gradients for the variable
            q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach()).squeeze()
            q_value_loss1 = (q_value_loss1).mean()
            q_value_loss2 = (q_value_loss2).mean()
        else:
            q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach()).squeeze()  # detach: no gradients for the variable
            q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach()).squeeze()
            td_error_abs = (q_value_loss1 + q_value_loss2)/2
            q_value_loss1 = (q_value_loss1 * weights).mean()
            q_value_loss2 = (q_value_loss2 * weights).mean()
            # update priority for trained samples

            self.replay_buffer.update_priorities(idxs, np.nan_to_num(td_error_abs.cpu().detach()) + 1e-6)


        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        if(self.use_grad_clip):
            nn.utils.clip_grad_norm_(self.soft_q_net1.parameters(),self.max_grad)
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        if(self.use_grad_clip):
            nn.utils.clip_grad_norm_(self.soft_q_net2.parameters(),self.max_grad)
        self.soft_q_optimizer2.step()  

        predicted_new_q_value = torch.min(self.soft_q_net1(cam_tensor,lidar_tensor, new_action, reward),self.soft_q_net2(cam_tensor,lidar_tensor, new_action, reward))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_net.version += 1

        self.q1_losses.append(q_value_loss1.item())
        self.q2_losses.append(q_value_loss2.item())
        self.policy_losses.append(policy_loss.item())

        
    
        if(self.use_hard_update):
            if (self.update_step % self.hard_update_n_step == 0) :
                for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
                    target_param.data.copy_(param.data)
                for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
                    target_param.data.copy_(param.data)


        else: # Soft update the target value net
            for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
                target_param.data.copy_(  # copy data value into target parameters
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )
            for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
                target_param.data.copy_(  # copy data value into target parameters
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )
            
        del cam_tensor, lidar_tensor, next_cam_tensor, next_lidar_tensor, action, prev_action, reward, prev_reward, done  
        return predicted_new_q_value.mean()

    def save_checkpoint(self, path, suffix):
        torch.save({
                    'config'            : self.save_config,
                    'update_step'       : self.update_step,
                    'soft_q_net1'       : self.soft_q_net1.state_dict(),
                    'soft_q_net2'       : self.soft_q_net2.state_dict(),
                    'policy_net'        : self.policy_net.state_dict(),
                    'log_alpha'         : self.log_alpha,
                    'soft_q_optimizer1' : self.soft_q_optimizer1.state_dict(),
                    'soft_q_optimizer2' : self.soft_q_optimizer2.state_dict(),
                    'policy_optimizer'  : self.policy_optimizer.state_dict(),
                    'alpha_optimizer'   : self.alpha_optimizer.state_dict()
                    }, path+"_CHECKPOINT_"+str(suffix))

    def load_checkpoint(self,path,suffix, load_config=False):
        checkpoint=torch.load(path+"_CHECKPOINT_"+str(suffix))

        self.soft_q_net1.load_state_dict(checkpoint['soft_q_net1'])
        self.soft_q_net2.load_state_dict(checkpoint['soft_q_net2'])
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.soft_q_optimizer1.load_state_dict(checkpoint['soft_q_optimizer1'])
        self.soft_q_optimizer2.load_state_dict(checkpoint['soft_q_optimizer2'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.log_alpha = checkpoint['log_alpha']
        print("ALHPA DEVICE!!!!!111!!!: ", self.log_alpha.get_device())
        self.update_step = checkpoint['update_step']
        if(load_config):
            self.save_config = checkpoint['config']
            self.load_config(checkpoint['config'])

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)
        print("LOADED PRETRAINED MODEL FROM: \n", path)
        print("NOW START TRAINING FROM STEP: ", self.update_step)

        self.all_elements_to(train_device)

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path+'_q1_'+str(self.update_step))
        torch.save(self.soft_q_net2.state_dict(), path+'_q2_'+str(self.update_step))
        torch.save(self.policy_net.state_dict(), path+'_policy_'+str(self.update_step))

    def load_model(self, path, step):
        self.soft_q_net1.load_state_dict(torch.load(path+'_q1_'+str(step)))
        self.soft_q_net2.load_state_dict(torch.load(path+'_q2_'+str(step)))
        self.policy_net.load_state_dict(torch.load(path+'_policy_'+str(step)))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()


        
"""

class SAC_Trainer_Resnet():
    def __init__(self,save_location, config):
        save_location           = save_location
        self.q1_losses          = []
        self.q2_losses          = []
        self.policy_losses      = []
        self.alpha_losses       = []
        self.alpha_list         = []
        self.update_step        = 0
        action_range            = 1.
        tau                     = config['tau']
        self.alpha              = config['alpha']
        q_lr                    = config['q_lr']
        p_lr                    = config['p_lr']
        a_lr                    = config['a_lr']
        buffer_maxlen           = config['buffer_maxlen']
        num_stacked_frames      = config['num_stacked_frames']
        hidden_dim              = config['hidden_dim']
        freeze_convolution      = config['freeze_convolution']
        self.use_grad_clip      = config['use_grad_clip']
        self.max_grad           = config['max_grad']
        self.use_exp_prio_buffer= config['use_exp_prio_buffer']
        use_pretrained_resnet18 = config['use_pretrained_resnet18']
        num_lidar_beams         = config['num_lidar_beams']
        action_dim              = 2
        input_shape             = config['output_resolution']
        input_shape             = (input_shape[0],input_shape[1],input_shape[2])
        state_dim               = (num_stacked_frames*input_shape[0],input_shape[1],input_shape[2],num_lidar_beams)
        
        

        self.soft_q_net1 = SoftQNetwork_Resnet(state_dim, action_dim, hidden_dim,freeze_convolution=freeze_convolution,use_pretrained_resnet18=use_pretrained_resnet18).to(device)
        self.soft_q_net2 = SoftQNetwork_Resnet(state_dim, action_dim, hidden_dim,freeze_convolution=freeze_convolution,use_pretrained_resnet18=use_pretrained_resnet18).to(device)
        self.target_soft_q_net1 = SoftQNetwork_Resnet(state_dim, action_dim, hidden_dim,freeze_convolution=freeze_convolution,use_pretrained_resnet18=use_pretrained_resnet18).to(device)
        self.target_soft_q_net2 = SoftQNetwork_Resnet(state_dim, action_dim, hidden_dim,freeze_convolution=freeze_convolution,use_pretrained_resnet18=use_pretrained_resnet18).to(device)
        self.policy_net = PolicyNetwork_Resnet(state_dim, action_dim, hidden_dim, action_range,freeze_convolution=freeze_convolution,use_pretrained_resnet18=use_pretrained_resnet18).to(device)
        self.log_alpha = torch.ones(1, device=device) * np.log(self.alpha)
        self.log_alpha.requires_grad=True


        if(self.use_exp_prio_buffer):
            self.replay_buffer = PrioritizedBuffer(buffer_maxlen,env.observtion_shape,memory_efficient_mode = False)
            self.soft_q_criterion1 = torch.nn.SmoothL1Loss(reduction = 'none')
            self.soft_q_criterion2 = torch.nn.SmoothL1Loss(reduction = 'none')
        else:
            self.replay_buffer = ReplayBuffer_MemEfficient_Resnet(buffer_maxlen,num_stacked_frames,state_dim)
            self.soft_q_criterion1 = nn.MSELoss()
            self.soft_q_criterion2 = nn.MSELoss()


        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=p_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=a_lr)

    
    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99,soft_tau=1e-2, use_reward_scaling=False):
        start = current_milli_time()
        if self.use_exp_prio_buffer == False:
            state, action,prev_action, reward, prev_reward, next_state, done = self.replay_buffer.sample(batch_size)
        else:
            transitions, idxs, weights = self.replay_buffer.sample(batch_size)
            state, action, prev_action, reward, next_state, done = transitions
            weights = torch.FloatTensor(weights).to(device)

        state_shape         =  state.shape
        cam_state           = []
        lidar_state         = []
        next_cam_state      = []
        next_lidar_state    = []

        for x in range(batch_size): 
            cam_state.append(state[x][0])
            lidar_state.append(state[x][1])
            next_cam_state.append(state[x][0])
            next_lidar_state.append(state[x][1])


        cam_state           = np.asarray(cam_state).reshape((batch_size,3,224,224))
        lidar_state         = np.asarray(lidar_state)

        cam_tensor          = torch.FloatTensor(cam_state).to(device)
        lidar_tensor        = torch.FloatTensor(lidar_state).to(device)

        next_cam_state      = np.asarray(next_cam_state).reshape((batch_size,3,224,224))
        next_lidar_state    = np.asarray(next_lidar_state)
        next_cam_tensor     = torch.FloatTensor(next_cam_state).to(device)
        next_lidar_tensor   = torch.FloatTensor(next_lidar_state).to(device)


        self.update_step +=1 

        state               = [cam_tensor,lidar_tensor]

        next_state          = [next_cam_tensor, next_lidar_tensor]

        action              = torch.FloatTensor(action).to(device)
        prev_action         = torch.FloatTensor(prev_action).to(device)
        reward              = torch.FloatTensor(reward).unsqueeze(1).to(device)       # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        prev_reward         = torch.FloatTensor(prev_reward).to(device)  # prev_reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done                = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
        
        

        predicted_q_value1 = self.soft_q_net1(state, action,prev_action, prev_reward)
        predicted_q_value2 = self.soft_q_net2(state, action,prev_action, prev_reward)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state,prev_action, prev_reward,device=device)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state,action, reward,device=device)

        if(use_reward_scaling):
            reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

    # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

            self.alpha_losses.append(alpha_loss.item())
            self.alpha_list.append(self.alpha.item())

        else:
            self.alpha = 1.
            alpha_loss = 0

    # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action, action, reward),self.target_soft_q_net2(next_state, new_next_action,action,reward)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
        # print("pred Q", predicted_q_value1)
        # print("target Q", target_q_value)


        if self.use_exp_prio_buffer == False:            
            q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
            q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())
        else:
            q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach()).squeeze()  # detach: no gradients for the variable
            q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach()).squeeze()
            # print("q1 loss", q_value_loss1)
            # print(predicted_q_value1.shape, target_q_value.shape)
            # print(weights.shape, self.soft_q_criterion1(predicted_q_value1, target_q_value.detach()).squeeze().shape)
            td_error_abs = (q_value_loss1 + q_value_loss2)/2
            q_value_loss1 = (q_value_loss1 * weights).mean()
            q_value_loss2 = (q_value_loss2 * weights).mean()
            # print("IDXS: ", idxs)
            # update priority for trained samples
            for idx, td_error in zip(idxs, td_error_abs.cpu().detach() + 1e-8):#in zip(idxs, td_error_abs.cpu().detach().numpy()[0]):
                self.replay_buffer.update_priority(idx, td_error)

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        if(self.use_grad_clip):
            nn.utils.clip_grad_norm_(self.soft_q_net1.parameters(),self.max_grad)
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        if(self.use_grad_clip):
            nn.utils.clip_grad_norm_(self.soft_q_net2.parameters(),self.max_grad)
        self.soft_q_optimizer2.step()  


        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action,action, reward),self.soft_q_net2(state, new_action,action, reward))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.q1_losses.append(q_value_loss1.item())
        self.q2_losses.append(q_value_loss2.item())
        self.policy_losses.append(policy_loss.item())
        
    # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return predicted_new_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path+'_q1_'+str(self.update_step))
        torch.save(self.soft_q_net2.state_dict(), path+'_q2_'+str(self.update_step))
        torch.save(self.policy_net.state_dict(), path+'_policy_'+str(self.update_step))

    def load_model(self, path, step):
        self.soft_q_net1.load_state_dict(torch.load(path+'_q1_'+str(step)))
        self.soft_q_net2.load_state_dict(torch.load(path+'_q2_'+str(step)))
        self.policy_net.load_state_dict(torch.load(path+'_policy_'+str(step)))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()

"""