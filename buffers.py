# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 23:24:08 2020

@author: hongh
"""
import numpy as np
from collections import deque
import random
import torch
import matplotlib.pyplot as plt
from utils import *
from segment_tree import * 

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, prev_action, reward, prev_reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action,prev_action, reward, prev_reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action,prev_action, reward, prev_reward, next_state, done = map(np.stack, zip(*batch)) # stack for each element
        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return state, action, prev_action, reward, prev_reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class ReplayBuffer_MemEfficient:
    def __init__(self, capacity,num_stacked_frames,observation_shape):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.not_allowed_idx = set(np.arange(num_stacked_frames))
        self.num_stacked_frames = num_stacked_frames
        self.stacked_images_shape    = (observation_shape[0],observation_shape[1],observation_shape[2])
    
    def add(self, obs_cam, obs_lidar, action, prev_action, reward, prev_reward, next_obs_cam, next_obs_lidar, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (obs_cam, obs_lidar, action,prev_action, reward, prev_reward, next_obs_cam, next_obs_lidar, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        
        # sample without replacement and avoid the first n=num_stacked_frames entries in the buffer to avoid: idx < frame_stack
        idx_list = np.random.choice(len(self.buffer)-self.num_stacked_frames, batch_size)+self.num_stacked_frames 


        # find terminating samples inside the batch
        for idx in range(len(idx_list)): 
            for frame in range(self.num_stacked_frames-1):
                obs_cam, obs_lidar  , action  , prev_action  , reward  , prev_reward  , next_obs_cam, next_obs_lidar  , done   = self.buffer[idx_list[idx]-frame]
                #if terminating frame is found in one of the 4 previous frames then set the index of the terminating frame to be the last one in that state
                #this avoids training on overlapping episodes, consider: the sampled idx is 13 -> then we concatenate the frames like so: [10,11,12,13] such that 13 is the last one
                # but now if frame 11 corresponds to a terminating state then frames 10 and 11 are in a different episode than 12 and 13 therefore a state containing the frames
                # [10,11,12,13] is impossible to reach (it would contain a teleport), therefore instead we find the done flag at frame 11 and reassign the target index from 13 to eleven
                # the resulting sample is now [8,9,10,11] instead of [10,11,12,13]
                if(done):
                    idx_list[idx] = idx_list[idx]-frame
                    break

        state_cam_batch = []
        next_cam_state_batch = []
        # stack the corresponding frames according to the idx_list and save them as batches
        ##### TODO: THIS TAKES TO LONG
        for idx in idx_list:
            obs_cam  , obs_lidar  , action  , prev_action  , reward  , prev_reward  , next_obs_cam  , next_obs_lidar  , done   = self.buffer[idx]
            obs_cam_1, obs_lidar_1, action_1, prev_action_1, reward_1, prev_reward_1, next_obs_cam_1, next_obs_lidar_1, done_1 = self.buffer[idx-1]
            obs_cam_2, obs_lidar_2, action_2, prev_action_2, reward_2, prev_reward_2, next_obs_cam_2, next_obs_lidar_2, done_2 = self.buffer[idx-2]
            obs_cam_3, obs_lidar_3, action_3, prev_action_3, reward_3, prev_reward_3, next_obs_cam_3, next_obs_lidar_3, done_3 = self.buffer[idx-3]
            obs_shape       = obs_cam.shape

            state_cam           = np.asarray([obs_cam_3, obs_cam_2, obs_cam_1, obs_cam]).reshape(self.stacked_images_shape)
            next_state_cam      = np.asarray([obs_cam_2, obs_cam_1, obs_cam, next_obs_cam]).reshape(self.stacked_images_shape)



            state_cam_batch.append(state_cam)
            next_cam_state_batch.append(next_state_cam)
            
            del obs_cam  , obs_lidar  , action  , prev_action  , reward  , prev_reward  , next_obs_cam  , next_obs_lidar  , done  
            del obs_cam_1, obs_lidar_1, action_1, prev_action_1, reward_1, prev_reward_1, next_obs_cam_1, next_obs_lidar_1, done_1
            del obs_cam_2, obs_lidar_2, action_2, prev_action_2, reward_2, prev_reward_2, next_obs_cam_2, next_obs_lidar_2, done_2
            del obs_cam_3, obs_lidar_3, action_3, prev_action_3, reward_3, prev_reward_3, next_obs_cam_3, next_obs_lidar_3, done_3



        batch = [self.buffer[i] for i in idx_list]


        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        obs_cam  , obs_lidar  , action  , prev_action  , reward  , prev_reward  , next_obs_cam  , next_obs_lidar  , done = map(np.stack, zip(*batch)) # stack for each element
        #replace the observations with the stacked states
        state_cam       = np.asarray(state_cam_batch)
        next_state_cam  = np.asarray(next_cam_state_batch)



        return state_cam,obs_lidar, action, prev_action, reward, prev_reward, next_state_cam, next_obs_lidar, done
    
    def __len__(self):
        return len(self.buffer)

    def show_state(self,state_tensor, save=False):
        state_shape= (84,84,3)
        pic0 = np.array(state_tensor.reshape(state_shape))    
        
        fig, (ax1) = plt.subplots(1, 1)
        ax1.imshow(pic0)
        plt.show()
  
class ReplayBuffer_MemEfficient_Resnet:
    def __init__(self, capacity,num_stacked_frames,observation_shape):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.not_allowed_idx = set(np.arange(num_stacked_frames))
        self.num_stacked_frames = num_stacked_frames
        self.stacked_images_shape    = (observation_shape[0],244,244)
    
    def push(self, observation, action, prev_action, reward, prev_reward, next_observation, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (observation, action,prev_action, reward, prev_reward, next_observation, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        
        # sample without replacement and avoid the first n=num_stacked_frames entries in the buffer to avoid: idx < frame_stack
        idx_list = np.random.choice(len(self.buffer)-self.num_stacked_frames, batch_size)+self.num_stacked_frames 


        # find terminating samples inside the batch
        for idx in range(len(idx_list)): 
            for frame in range(self.num_stacked_frames-1):
                observation  , action  , prev_action  , reward  , prev_reward  , next_observation  , done   = self.buffer[idx_list[idx]-frame]
                #if terminating frame is found in one of the 4 previous frames then set the index of the terminating frame to be the last one in that state
                #this avoids training on overlapping episodes, consider: the sampled idx is 13 -> then we concatenate the frames like so: [10,11,12,13] such that 13 is the last one
                # but now if frame 11 corresponds to a terminating state then frames 10 and 11 are in a different episode than 12 and 13 therefore a state containing the frames
                # [10,11,12,13] is impossible to reach (it would contain a teleport), therefore instead we find the done flag at frame 11 and reassign the target index from 13 to eleven
                # the resulting sample is now [8,9,10,11] instead of [10,11,12,13]
                if(done):
                    idx_list[idx] = idx_list[idx]-frame
                    break

        state_batch = []
        next_state_batch = []
        # stack the corresponding frames according to the idx_list and save them as batches
        for idx in idx_list:
            observation  , action  , prev_action  , reward  , prev_reward  , next_observation  , done   = self.buffer[idx]
            # observation_1, action_1, prev_action_1, reward_1, prev_reward_1, next_observation_1, done_1 = self.buffer[idx-1]
            # observation_2, action_2, prev_action_2, reward_2, prev_reward_2, next_observation_2, done_2 = self.buffer[idx-2]
            # observation_3, action_3, prev_action_3, reward_3, prev_reward_3, next_observation_3, done_3 = self.buffer[idx-3]
            obs_shape       = observation[0].shape

            state_lidar         = observation[1]
                                    #preprocess(observation_3[0]).numpy(),preprocess(observation_2[0]).numpy(),preprocess(observation_1[0]).numpy(),
            state_cam           = [preprocess(observation[0]).numpy()]
            state_cam           = np.asarray(state_cam).reshape((3,224,224))
            state               = np.asarray([state_cam, state_lidar])

            next_state_lidar    = next_observation[1]
                #preprocess(observation_2[0]).numpy(),preprocess(observation_1[0]).numpy(),preprocess(observation[0]).numpy(),
            next_state_cam = [preprocess(next_observation[0]).numpy()]
            next_state_cam = np.asarray(next_state_cam).reshape((3,224,224))
            next_state          = np.asarray([next_state_cam,next_state_lidar])

            state_batch.append(state)
            next_state_batch.append(next_state)


        batch = [self.buffer[i] for i in idx_list]


        ''' 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        observation, action,prev_action, reward, prev_reward, next_observation, done = map(np.stack, zip(*batch)) # stack for each element
        #replace the observations with the stacked states
        state       = np.asarray(state_batch)
        next_state  = np.asarray(next_state_batch)


        return state, action, prev_action, reward, prev_reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

    def show_state(self,state_tensor, save=False):
        state_shape= (84,84,1)
        pic0 = np.array(state_tensor.reshape(state_shape))    
        
        fig, (ax1) = plt.subplots(1, 1)
        ax1.imshow(pic0)
        plt.show()

class BasicBuffer:
    def __init__(self, max_size, shape):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)       
        self.frame_stack = shape[0]
        self.not_allowed_idx = set(np.arange(self.frame_stack))
        self.img_shape = shape
        
    def push(self, state, action, reward, next_state, done):
        ''' Changed to store the last frame'''
        # state = np.expand_dims(state[-1], axis=0)
        # next_state = np.expand_dims(next_state[-1], axis=0)
        # state.astype(np.uint8)
        # next_state.astype(np.uint8)
        experience = (np.expand_dims(state[-1], axis=0).astype(np.uint8), np.array([action]), np.array([reward]), np.expand_dims(next_state[-1], axis=0).astype(np.uint8), done)
        self.buffer.append(experience)
        

    def sample(self, batch_size):
        '''Changed to memory-efficient one by storing one frame, but when extracting, frame stacking is automatically done.'''
        # state_batch = []
        action_batch = []
        reward_batch = []
        # next_state_batch = []
        done_batch = []
        continue_sample = True
        state_batch = np.zeros((batch_size,*self.img_shape))
        next_state_batch = np.zeros((batch_size,*self.img_shape))
        
        # ----------------recursive sampling to avoid: idx < frame_stack----------------
        while continue_sample:
            idx = np.random.choice(len(self.buffer), batch_size) # sample without replacement
            idx_set = set(idx)
            if len(idx_set.intersection(self.not_allowed_idx)) == 0:
                continue_sample = False          

                  
        # --------------- frame stacking -----------------         
        for i in range(len(idx)):
            for j in range(1, self.frame_stack+1):# 
                if self.buffer[idx[i]-j+1][4] == True and j > 1: # done = True         
                    state_batch[i, self.frame_stack-j] = state_batch[i, self.frame_stack-j+1]  # the last [0] makes --> 1*84*84 --> 84*84 
                    next_state_batch[i, self.frame_stack-j] = state_batch[i, self.frame_stack-j+1]
                else:
                    state_batch[i, self.frame_stack-j] = self.buffer[idx[i]-j+1][0][0]  # the last [0] makes --> 1*84*84 --> 84*84 
                    next_state_batch[i, self.frame_stack-j] = self.buffer[idx[i]-j+1][3][0]
                # state_batch.append(self.buffer[idx[i]][0])
                
                # next_state_batch.append(self.buffer[idx[i]][3])
                
        for i in range(len(idx)):
            reward_batch.append(self.buffer[idx[i]][2])
            action_batch.append(self.buffer[idx[i]][1])
            done_batch.append(self.buffer[idx[i]][4])
            
        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)
    
    
    
class SumTree:
    '''Even if I can't apply the trick of deque here, it is still possible to store one frame by ruling out the index in the current write position.'''
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=object) # storing experience
        self.tree = np.zeros(2 * capacity - 1) # storing pripority + parental nodes. If you have n elements in the bottom, then you need n + (n-1) nodes to construct a tree.        
        self.n_entries = 0
        self.overwrite_start_flag = False # record whether N_entry > capacity
        
    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])


    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1 # starting write from the first element of the bottom layer,
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            self.overwrite_start_flag = True
        if self.n_entries < self.capacity:
            self.n_entries += 1


    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        #--------original----------
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])   
        #--------The below is Morvan's implementation----------
        #https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Dueling%20Double%20DQN%20with%20PER%20and%20fixed-q%20targets/.ipynb_checkpoints/Dueling%20Deep%20Q%20Learning%20with%20Doom%20(%2B%20double%20DQNs%20and%20Prioritized%20Experience%20Replay)-checkpoint.ipynb
        #https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
#        parent_idx = 0
#        while True:     # the while loop is faster than the method in the reference code
#            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
#            cr_idx = cl_idx + 1
#            if cl_idx >= len(self.tree):        # reach bottom, end search
#                leaf_idx = parent_idx
#                break
#            else:       # downward search, always search for a higher priority node
#                if s <= self.tree[cl_idx]:
#                    parent_idx = cl_idx
#                else:
#                    s -= self.tree[cl_idx]
#                    parent_idx = cr_idx
#        data_idx = leaf_idx - self.capacity + 1
#        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]   
    
    
    def max_priority(self):
        return np.max(self.tree[self.capacity-1:])

    # ---- modified : later to avoid min_prob being overwritten ----
    def min_prob(self):
        if self.overwrite_start_flag == False:        
            p_min = float(np.min(self.tree[self.capacity-1:self.n_entries+self.capacity-1])) / self.total()
            self.p_min_history = p_min
        elif self.overwrite_start_flag == True: 
            p_min = min( float(np.min(self.tree[self.capacity-1:self.n_entries+self.capacity-1])) / self.total() ,self.p_min_history)
            self.p_min_history = min(p_min, self.p_min_history)
        return p_min
    
    
    
class PrioritizedBuffer:
    '''Currently, there is a problem when memory is full'''
    def __init__(self, max_size, shape, alpha=0.6, beta=0.4, memory_efficient_mode = True):
        self.sum_tree = SumTree(max_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta)/100000.
        self.current_length = 0
        self.frame_stack = shape[0]
        self.efficient_memory_mode = memory_efficient_mode
        self.img_shape = shape
        
        
    def push(self, state, action,prev_action, reward, next_state, done):
        priority = 1.0 if self.current_length == 0 else self.sum_tree.max_priority()
        self.current_length = self.current_length + 1
        # priority = priority ** self.alpha
        # -------------modified for efficient storage--------------
        if self.efficient_memory_mode:
            experience = (np.expand_dims(state[-1], axis=0).astype(np.uint8), np.array([action]), np.array([reward]), np.expand_dims(next_state[-1], axis=0).astype(np.uint8), done)
        else:
            experience = (state, action,prev_action, reward, next_state, done)
        self.sum_tree.add(priority, experience)


    def sample(self, batch_size):
        # ----------------To modify: sampling for efficient memory storage------------------
        batch_idx, batch, priorities = [], [], []
        segment = self.sum_tree.total() / batch_size
        p_sum = self.sum_tree.tree[0]

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            # ---------self modified to allow storing for efficient memory storage of only one frame----------
            if self.efficient_memory_mode:
                while True:
                    s = random.uniform(a, b)
                    idx, p, data = self.sum_tree.get(s)
                    data_idx = idx-self.sum_tree.capacity+1 # Note: idx(node idx) is different from dataidx (data idx)!
                    # print(data_idx, self.sum_tree.write, self.frame_stack)
                    if (data_idx - self.sum_tree.write > self.frame_stack or data_idx<= self.sum_tree.write) and (self.sum_tree.overwrite_start_flag == True): 
                        break
                    # elif (self.sum_tree.overwrite_start_flag == False) and (data_idx >= self.frame_stack - 1):
                    #     break         
                    elif (self.sum_tree.overwrite_start_flag == False): # allow samples from <= frame_stack
                        break                                   
            else:
                s = random.uniform(a, b)
                idx, p, data = self.sum_tree.get(s)
            batch_idx.append(idx)
            batch.append(data)
            priorities.append(p)
        prob = np.array(priorities) /p_sum 
        IS_weights = np.power(self.sum_tree.n_entries * prob, -self.beta)
        max_weight = np.power(self.sum_tree.n_entries * self.sum_tree.min_prob(), -self.beta)
        IS_weights /= max_weight
        
        if self.efficient_memory_mode:
            action_batch = []
            reward_batch = []
            done_batch = []
            state_batch = np.zeros((batch_size,*self.img_shape))
            next_state_batch = np.zeros((batch_size,*self.img_shape))
            # --------------- frame stacking -----------------         
            for i in range(len(batch_idx)):
                for j in range(1, self.frame_stack+1):
                    data_idx = batch_idx[i]-self.sum_tree.capacity+1-j+1 # notice the offset to substract to get right data[data_idx]
                    if data_idx < 0 and self.sum_tree.overwrite_start_flag == True: # for the case : sample the 1st element and stack the last 3 elements from the end
                        data_idx += self.sum_tree.capacity
                    elif data_idx < 0 and self.sum_tree.overwrite_start_flag == False: # for the case : sample the 2nd element and stack the 1st elements 3 times from the beginning
                        data_idx = 0
                    # print(data_idx ,self.sum_tree.data[data_idx])
                    if (self.sum_tree.data[data_idx][4] == True and j > 1) or (data_idx < 0): # for the case of done = True or sampling from the beginning         
                        state_batch[i, self.frame_stack-j] = state_batch[i, self.frame_stack-j+1]  # the last [0] makes --> 1*84*84 --> 84*84 
                        next_state_batch[i, self.frame_stack-j] = state_batch[i, self.frame_stack-j+1]
                    else:
                        state_batch[i, self.frame_stack-j] = self.sum_tree.data[data_idx][0][0]  # the last [0] makes --> 1*84*84 --> 84*84 
                        next_state_batch[i, self.frame_stack-j] = self.sum_tree.data[data_idx][3][0]
                 
            for i in range(len(batch_idx)):
                data_idx = batch_idx[i]-self.sum_tree.capacity+1
                reward_batch.append(self.sum_tree.data[data_idx][2])
                action_batch.append(self.sum_tree.data[data_idx][1])
                done_batch.append(self.sum_tree.data[data_idx][4])           
        else:
            state_batch = []
            action_batch = []
            prev_action_batch = []
            reward_batch = []
            next_state_batch = []
            done_batch = []
            for transition in batch:
                state, action, prev_action,reward, next_state, done = transition
                state_batch.append(state)
                action_batch.append(action)
                prev_action_batch.append(prev_action)
                reward_batch.append(reward)
                next_state_batch.append(next_state)
                done_batch.append(done)
        self.beta = min(1., self.beta + self.beta_increment)
        return (state_batch, action_batch,prev_action_batch, reward_batch, next_state_batch, done_batch), batch_idx, IS_weights

    def update_priority(self, idx, td_error):
        priority =  np.power(td_error, self.alpha)
        self.sum_tree.update(idx, priority)

    def __len__(self):
        return self.current_length
    
    
    
class PrioritizedBuffer_SNAIL:   
    '''Only designed for TCN mode'''
    def __init__(self, max_size, obs_shape, action_n, alpha=0.6, beta=0.4):
        self.sum_tree = SumTree(max_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta)/100000.
        self.current_length = 0
        self.frame_stack = obs_shape[0]
        self.efficient_memory_mode = True
        self.img_shape = obs_shape
        self.action_n = action_n
        
    def push(self, state, action, reward, next_state, done):
        priority = 1.0 if self.current_length == 0 else self.sum_tree.max_priority()
        self.current_length = self.current_length + 1
        #priority = td_error ** self.alpha
        # -------------modified for efficient storage--------------
        if self.efficient_memory_mode:
            experience = (np.expand_dims(state[-1], axis=0).astype(np.uint8), np.array([action]), np.array([reward]), np.expand_dims(next_state[-1], axis=0).astype(np.uint8), done)
        else:
            experience = (state, action, reward, next_state, done)
        self.sum_tree.add(priority, experience)


    def sample(self, batch_size):
        # ----------------To modify: sampling for efficient memory storage------------------
        batch_idx, batch, priorities = [], [], []
        segment = self.sum_tree.total() / batch_size
        p_sum = self.sum_tree.tree[0]

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            # ---------self modified to allow storing for efficient memory storage of only one frame----------
            if self.efficient_memory_mode:
                while True:
                    s = random.uniform(a, b)
                    idx, p, data = self.sum_tree.get(s)
                    data_idx = idx-self.sum_tree.capacity+1 # Note: idx(node idx) is different from dataidx (data idx)!
                    # print(data_idx, self.sum_tree.write, self.frame_stack)
                    if (data_idx - self.sum_tree.write > self.frame_stack or data_idx<= self.sum_tree.write) and (self.sum_tree.overwrite_start_flag == True): 
                        break
                    # elif (self.sum_tree.overwrite_start_flag == False) and (data_idx >= self.frame_stack - 1):
                    #     break         
                    elif (self.sum_tree.overwrite_start_flag == False): # allow samples from <= frame_stack
                        break                                   
            else:
                s = random.uniform(a, b)
                idx, p, data = self.sum_tree.get(s)
            batch_idx.append(idx)
            batch.append(data)
            priorities.append(p)
        prob = np.array(priorities) /p_sum 
        IS_weights = np.power(self.sum_tree.n_entries * prob, -self.beta)
        max_weight = np.power(self.sum_tree.n_entries * self.sum_tree.min_prob(), -self.beta)
        IS_weights /= max_weight
        
        if self.efficient_memory_mode:
            # action: (B,L*feat), 
            # reward: (B,L*1)
            action_batch_s = np.zeros((batch_size, self.action_n, self.img_shape[0])) 
            reward_batch_s = np.zeros((batch_size, 1, self.img_shape[0])) 
            reward_batch = []
            action_batch = []
            done_batch = []
            state_batch = np.zeros((batch_size,*self.img_shape))
            next_state_batch = np.zeros((batch_size,*self.img_shape))
            # --------------- frame stacking -----------------         
            for i in range(len(batch_idx)):
                for j in range(1, self.frame_stack+1):
                    data_idx = batch_idx[i]-self.sum_tree.capacity+1-j+1 # notice the offset to substract to get right data[data_idx]
                    if data_idx < 0 and self.sum_tree.overwrite_start_flag == True: # for the case : sample the 1st element and stack the last 3 elements from the end
                        data_idx += self.sum_tree.capacity
                    elif data_idx < 0 and self.sum_tree.overwrite_start_flag == False: # for the case : sample the 2nd element and stack the 1st elements 3 times from the beginning
                        data_idx = 0
                    # print(data_idx ,self.sum_tree.data[data_idx])
                    if (self.sum_tree.data[data_idx][4] == True and j > 1) or (data_idx < 0): # for the case of done = True or sampling from the beginning         
                        state_batch[i, self.frame_stack-j] = state_batch[i, self.frame_stack-j+1]  # the last [0] makes --> 1*84*84 --> 84*84 
                        next_state_batch[i, self.frame_stack-j] = state_batch[i, self.frame_stack-j+1]
                        action_batch_s[i,:,self.frame_stack-j] = np.array([0]*self.action_n)# action_batch_s[i, self.frame_stack-j+1]
                        reward_batch_s[i,:,self.frame_stack-j] = np.array([0])# reward_batch_s[i, self.frame_stack-j+1]
                    else:
                        state_batch[i, self.frame_stack-j] = self.sum_tree.data[data_idx][0][0]  # the last [0] makes --> 1*84*84 --> 84*84 
                        next_state_batch[i, self.frame_stack-j] = self.sum_tree.data[data_idx][3][0]
                        action_batch_s[i,:,self.frame_stack-j] = self.sum_tree.data[data_idx][1][0]
                        reward_batch_s[i,:,self.frame_stack-j] = self.sum_tree.data[data_idx][2][0]
                 
            for i in range(len(batch_idx)):
                data_idx = batch_idx[i]-self.sum_tree.capacity+1
                reward_batch.append(self.sum_tree.data[data_idx][2])
                action_batch.append(self.sum_tree.data[data_idx][1])
                done_batch.append(self.sum_tree.data[data_idx][4])           
        else:
            state_batch = []
            action_batch = []
            reward_batch = []
            next_state_batch = []
            done_batch = []
            for transition in batch:
                state, action, reward, next_state, done = transition
                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)
                next_state_batch.append(next_state)
                done_batch.append(done)
        self.beta = min(1., self.beta + self.beta_increment)
        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch, action_batch_s, reward_batch_s), batch_idx, IS_weights

    def update_priority(self, idx, td_error):
        priority =  np.power(td_error, self.alpha)
        self.sum_tree.update(idx, priority)

    def __len__(self):
        return self.current_length    
    
    
    
class PrioritizedBuffer_N_step:
    '''The difference to is gamma_exp is another element saved
    '''
    def __init__(self, max_size, shape, alpha=0.6, beta=0.4):
        self.sum_tree = SumTree(max_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta)/100000.
        self.current_length = 0
        self.frame_stack = shape[0]
        self.efficient_memory_mode = True
        self.img_shape = shape
        
        
    def push(self, state, action, reward, next_state, done, gamma_exp):
        priority = 1.0 if self.current_length == 0 else self.sum_tree.max_priority()
        self.current_length = self.current_length + 1
        #priority = td_error ** self.alpha
        # -------------modified for efficient storage--------------
        if self.efficient_memory_mode:
            experience = (np.expand_dims(state[-1], axis=0).astype(np.uint8), np.array([action]), np.array([reward]), np.expand_dims(next_state[-1], axis=0).astype(np.uint8), done, np.array([gamma_exp]))
        else:
            experience = (state, np.array([action]), np.array([reward]), next_state, done, np.array([gamma_exp]))
        self.sum_tree.add(priority, experience)


    def sample(self, batch_size):
        # ----------------To modify: sampling for efficient memory storage------------------
        batch_idx, batch, priorities = [], [], []
        segment = self.sum_tree.total() / batch_size
        p_sum = self.sum_tree.tree[0]
    
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            # ---------self modified to allow storing for efficient memory storage of only one frame----------
            if self.efficient_memory_mode:
                while True:

                    s = random.uniform(a, b)
                    idx, p, data = self.sum_tree.get(s)
                    data_idx = idx-self.sum_tree.capacity+1 # Note: idx(node idx) is different from dataidx (data idx)!
                    # print(data_idx, self.sum_tree.write, self.frame_stack)
                    if (data_idx - self.sum_tree.write > self.frame_stack or data_idx<= self.sum_tree.write) and (self.sum_tree.overwrite_start_flag == True): 
                        break
                    # elif (self.sum_tree.overwrite_start_flag == False) and (data_idx >= self.frame_stack - 1):
                    #     break         
                    elif (self.sum_tree.overwrite_start_flag == False): # allow samples from <= frame_stack
                        break                                            
            else:
                s = random.uniform(a, b)
                idx, p, data = self.sum_tree.get(s)
            batch_idx.append(idx)
            batch.append(data)
            priorities.append(p)
        prob = np.array(priorities) /p_sum 
        IS_weights = np.power(self.sum_tree.n_entries * prob, -self.beta)
        max_weight = np.power(self.sum_tree.n_entries * self.sum_tree.min_prob(), -self.beta)
        IS_weights /= max_weight
        
        if self.efficient_memory_mode:
            action_batch = []
            reward_batch = []
            done_batch = []
            state_batch = np.zeros((batch_size,*self.img_shape))
            next_state_batch = np.zeros((batch_size,*self.img_shape))
            gamma_exp_batch = []
            # --------------- frame stacking -----------------         
            for i in range(len(batch_idx)):
                for j in range(1, self.frame_stack+1):
                    data_idx = batch_idx[i]-self.sum_tree.capacity+1  -j+1 # notice the offset to substract to get right data[data_idx]
                    if data_idx < 0 and self.sum_tree.overwrite_start_flag == True: # for the case : sample the 1st element and stack the last 3 elements from the end
                        data_idx += self.sum_tree.capacity
                    elif data_idx < 0 and self.sum_tree.overwrite_start_flag == False: # for the case : sample the 2nd element and stack the 1st elements 3 times from the beginning
                        data_idx = 0
                    # print(data_idx ,self.sum_tree.data[data_idx])
                    if (self.sum_tree.data[data_idx][4] == True and j > 1) or (data_idx < 0): # for the case of done = True or sampling from the beginning         
                        state_batch[i, self.frame_stack-j] = state_batch[i, self.frame_stack-j+1]  # the last [0] makes --> 1*84*84 --> 84*84 
                        next_state_batch[i, self.frame_stack-j] = state_batch[i, self.frame_stack-j+1]
                    else:
                        state_batch[i, self.frame_stack-j] = self.sum_tree.data[data_idx][0][0]  # the last [0] makes --> 1*84*84 --> 84*84 
                        next_state_batch[i, self.frame_stack-j] = self.sum_tree.data[data_idx][3][0]
                 
            for i in range(len(batch_idx)):
                data_idx = batch_idx[i]-self.sum_tree.capacity+1
                reward_batch.append(self.sum_tree.data[data_idx][2])
                action_batch.append(self.sum_tree.data[data_idx][1])
                done_batch.append(self.sum_tree.data[data_idx][4])
                gamma_exp_batch.append(self.sum_tree.data[data_idx][5])
        else:
            state_batch = []
            action_batch = []
            reward_batch = []
            next_state_batch = []
            done_batch = []
            gamma_exp_batch = []
            for transition in batch:
                state, action, reward, next_state, done, gamma_exp = transition
                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)
                next_state_batch.append(next_state)
                done_batch.append(done)
                gamma_exp_batch.append(gamma_exp)
        self.beta = min(1., self.beta + self.beta_increment)
        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch, gamma_exp_batch), batch_idx, IS_weights

    def update_priority(self, idx, td_error):
        priority =  np.power(td_error, self.alpha)
        self.sum_tree.update(idx, priority)

    def __len__(self):
        return self.current_length
    

