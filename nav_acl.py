from utils import *
import matplotlib.pyplot as plt
from utils import rel_rot, rot_vec
from numpy import savetxt
import numpy as np
import scipy
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R
from scipy import stats, optimize, interpolate
import random
from Networks import NavACLNetwork, SoftQNetwork
from isaac_q_helper_functions import *
from RobotTask import RobotTask, Tasktype
import json
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from per_buffer import ReplayBuffer
import random
import time


class Nav_ACL():
  """
    My Interpretation of the Nav_ACL Algorithm for automatic Curriculum generation
    https://arxiv.org/abs/2009.05429
  """
  def __init__(self,nav_acl_network,config,worker_instance=True, shared_q_net=None, env=None, task_offset=np.array([0,0,0]), agent_index = 0, shared_elements=None):
    super(Nav_ACL, self).__init__()
    """
        my Implementaion keeps track of all previously generated tasks and their outcomes
    """
    self.config                         = config
    self.tasks                          = []
    self.steps                          = []
    self.rewards                        = []
    self.dones                          = []
    self.params                         = []
    self.default_task                   = self.config['default_task_omniverse']
    self.GOID_limits                    = self.config['GOID_limits']
    self.task_gen_method                = self.config['task_generation_method']
    self.initial_orientation            = np.array([1,0]) # it is assumed that the initial 2D orientation of the robot and the dolly is [1,0] and that the applied is relative to this initial orientation
    self.NavACL_loss_func               = torch.nn.BCELoss()
    self.adaptive_filtering_task_types  = [Tasktype.RANDOM, Tasktype.EASY, Tasktype.FRONTIER]
    self.adaptive_filtering_task_probs  = [0.15  ,   0.425   ,   0.425 ]
    self.device                         = train_device
    self.shared_elements                = shared_elements
    self.agent_index                    = agent_index
    self.q_net                          = SoftQNetwork(self.config)
    self.q_net.to(Qricculum_device)

    if(not worker_instance): # this instance of NavACL is the one used to train the network i.e. it does not receive copies ->  it produces the copies 
      self.NavACLNet        = nav_acl_network
      self.NavACL_optimizer = optim.Adam(self.NavACLNet.parameters(), lr=self.config['nav_acl_lr'])
      self.q_net = None
      self.env   = None
    else:
      if(self.config['q_ricculum_learning']):
        if(self.config['use_AF_database']): ## use a predefined database for adaptive filtering method of nav_acl (only relevant for qricculum learning, bc. it prevents generating tasks wich are only used for creating the adaptive boundaries)
          self.load_AF_database(self.config['AF_database_path']+str(agent_index)+".npy")
          print("using qricculum learning for agent: ", agent_index, " with task AF_database: ", self.config['AF_database_path']+str(agent_index)+".npy")
        assert (shared_q_net is not None)
        assert (env is not None)
        self.env = env
        self.update_networks(nav_acl_network,shared_q_net,task_offset)
      else:
        self.update_networks(nav_acl_network,q_net=None)
        self.NavACLNet = nav_acl_network

    if(self.config['using_unity']):
      self.default_task  = self.config['default_task_unity']
    else:
      self.default_task = self.config['default_task_omniverse']

  def update_networks(self,nav_acl_net, q_net, translation_offset=np.array([0,0,0])):
    if(self.config['q_ricculum_learning']):
        assert (q_net is not None)
        self.q_net.load_state_dict(q_net.state_dict())
        for param in self.q_net.parameters():
            param.requires_grad = False

    if(self.config['using_unity']):
      self.default_task  = self.config['default_task_unity']

    self.NavACLNet = NavACLNetwork(5,self.config['nav_acl_hidden_dim'])
    self.NavACLNet.load_state_dict(nav_acl_net.state_dict())
    self.NavACLNet.to(self.device)

  def get_task_difficulty_measure(self, robot_task):
    success_probability = self.get_task_success_probability(robot_task)
    return (1-success_probability)
  
  def get_task_success_probability(self, robot_task):
    task_params_array = self.get_task_params_array(robot_task,normalize=self.config['normalize_tasks'])
    task_params_array = torch.FloatTensor(task_params_array).to(self.device)
    return self.NavACLNet(task_params_array)

  def normalize_task_params_array(self, task_params_array):
    # task_params_array = (task_params_array - self.task_mean) / (self.deviser+1e-3)
    r_rot = self.config['randomization_params']['robot_randomization']['rotation_rnd'][0] / 2
    d_rot = self.config['randomization_params']['dolly_randomization']['rotation_rnd'][0] / 2
    rot   = r_rot + d_rot
    task_params_array[0] = np.interp(task_params_array[0], (self.config['randomization_params']['min_dist_dolly_robot'], self.config['randomization_params']['max_dist_dolly_robot']), (-1,1))
    task_params_array[1] = np.interp(task_params_array[1], (self.config['randomization_params']['min_dist_dolly_obs'], self.config['randomization_params']['max_dist_dolly_obs']), (-1,1))
    task_params_array[2] = np.interp(task_params_array[2], (self.config['randomization_params']['min_dist_robot_obs'], self.config['randomization_params']['max_dist_robot_obs']), (-1,1))
    task_params_array[3] = np.interp(task_params_array[3], (-(rot*np.pi)/180, (rot*np.pi)/180), (-1,1))
    task_params_array[4] = np.interp(task_params_array[4], (self.config['collision_penalty']-self.config['goal_reward'],self.config['goal_reward']), (-1,1))
    return task_params_array

  def get_task_params_array(self, task,normalize=True):
    if(self.config['q_ricculum_learning']):
      task_params_array = np.array([self.compute_dolly_robot_distance(task),
                                    self.compute_dolly_min_distance(task),
                                    self.compute_robot_min_distance(task),
                                    self.compute_relative_rotation_robot_dolly(task),
                                    np.clip(task.q_value,self.config['collision_penalty']-self.config['goal_reward'],self.config['goal_reward'])])
    else:
      task_params_array = np.array([self.compute_dolly_robot_distance(task),
                                    self.compute_dolly_min_distance(task),
                                    self.compute_robot_min_distance(task),
                                    self.compute_relative_rotation_robot_dolly(task),
                                    0.1])
    if normalize:
      task_params_array = self.normalize_task_params_array(task_params_array)
    return task_params_array

  def apply_offset_to_robot(self, robot_task, translation_offset):
    robot_task.robot_translation      =robot_task.robot_translation + translation_offset
    return robot_task
  

  def apply_offset_to_dolly(self, robot_task, translation_offset):
    robot_task.dolly_translation      += translation_offset
    return robot_task


  def create_randomized_task(self,translation_offset= np.array([0,0,0])):
    robot_task                        = RobotTask(self.default_task)
    robot_task.robot_translation      =robot_task.robot_translation + translation_offset
    robot_task.dolly_translation      += translation_offset
    robot_task.obstacle_translation   += translation_offset
    robot_task.obstacle_1_translation += translation_offset
    robot_task.obstacle_2_translation += translation_offset
    robot_task.obstacle_3_translation += translation_offset
    robot_task.randomize_task(self.config['randomization_params'])
    return robot_task



  def sample_random_task(self,translation_offset= np.array([0,0,0])):
    robot_task = self.create_randomized_task(translation_offset=translation_offset)

    while self.check_if_task_is_valid(robot_task) is not True:
      robot_task = self.create_randomized_task(translation_offset=translation_offset)

    if(self.config['q_ricculum_learning']):
      robot_task.q_value = self.get_q_value_for_task(robot_task,translation_offset)
      # print("q value of task: ", self.get_task_params_array(robot_task,False), " is : ", robot_task.q_value)
    return robot_task

  def save_tasks_for_later(self,translation_offset, num_tasks=1000, save_location="/home/developer/Training_results/Qricculum_Learning"):
    task_list             = []
    for task in range(num_tasks):
      robot_task            = self.create_randomized_task(translation_offset=translation_offset)
      obs_cam, obs_lidar    = self.env.reset(robot_task) 
      prev_action           = np.zeros((2))
      prev_reward           = [self.config['step_penalty']]
      task_array            = robot_task.get_task_array()
      task_list.append((obs_cam, obs_lidar,prev_action,prev_reward,task_array))
      if (task % 1000 == 0 ):
        print("saved so far: ", task)
    task_array = np.array(task_list)
    np.save(save_location,task_array)


  def load_AF_database(self, AF_database_path):
    """
      loads a task database from an .npy file and stores it as a list
    """
    self.AF_database = np.load(AF_database_path, allow_pickle=True)
    self.AF_database = self.AF_database.tolist() # this way we can use random.sample(list, num_elements)
    print("using qricculum learning with database: ", AF_database_path)
    self.dtb_stack = self.create_dtb_stack()


  def observation_to_q_value(self, obs_cam, obs_lidar, prev_action, prev_reward):
    n_s_f = self.config['num_stacked_frames']
    stacked_camera_obsrv  = deque(maxlen=n_s_f)
    transitions           = deque(maxlen=n_s_f)
    # start with a fresh state
    for i in range(self.config['num_stacked_frames']):
        stacked_camera_obsrv.append(obs_cam)
        if(self.config['use_snail_mode']):
            transitions.append((obs_cam, obs_lidar,prev_action,prev_reward))

    stacked_camera_state = np.asarray(stacked_camera_obsrv).reshape((self.env.observtion_shape[0]*n_s_f,self.env.observtion_shape[1],self.env.observtion_shape[2]))
    cam_tensor           = torch.FloatTensor(stacked_camera_state).unsqueeze(0).to(Qricculum_device)
    lidar_tensor         = torch.FloatTensor(obs_lidar).unsqueeze(0).to(Qricculum_device)
    prev_action          = torch.FloatTensor(prev_action).unsqueeze(0).to(Qricculum_device)
    reward               = torch.FloatTensor(prev_reward).unsqueeze(0).to(Qricculum_device)       # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
    with torch.no_grad():
      q_value = self.q_net(cam_tensor,lidar_tensor, prev_action, reward).detach().cpu().numpy().flatten()
      del stacked_camera_state, cam_tensor, lidar_tensor, prev_action, reward, obs_cam, obs_lidar, stacked_camera_obsrv
      return q_value[0]






  def get_q_value_for_task(self,robot_task,translation_offset):
    obs_cam, obs_lidar    = self.env.reset(robot_task) 
    prev_action           = np.zeros(8)
    prev_reward           = np.ones(4)*self.config['step_penalty']
    # self.show_state(obs_cam,True,string_extra=str(translation_offset))
    q                     = self.observation_to_q_value(obs_cam, obs_lidar,prev_action, prev_reward)
    return q

  def show_state(self,state_tensor, save=False, string_extra=""):
      state_shape= (84,84,3)
      state_tensor = state_tensor / 2**8
      pic0 = np.array(state_tensor.reshape(state_shape).astype(np.float32))    
      
      fig, (ax1) = plt.subplots(1, 1)
      ax1.imshow(pic0)
      if(save):
        path = "/home/developer/Pictures"+string_extra+str(current_milli_time())+".png"
        print("saving image as: ", path)
        plt.imsave(path,pic0)
      else:
        plt.show()

  def generate_random_task(self, translation_offset=np.array([0,0,0])):
    """
      returns a randomized RobotTask:
      
      based on a default task given in NAV_ACL_hyperparameters.json
    """
    if self.task_gen_method == "GOID":
      return self.generate_random_task_GOID(translation_offset)
    
    if self.task_gen_method == "AF":
      return self.generate_random_task_AF(translation_offset)



  def generate_random_task_GOID(self, translation_offset):
    while True:
      robot_task = self.sample_random_task(translation_offset)
      difficulty = self.get_task_difficulty_measure(robot_task).cpu()
      if ( (difficulty > self.GOID_limits['lower_limit'] ) and (difficulty < self.GOID_limits['upper_limit'])) :
        return robot_task

  def generate_random_task_AF(self, translation_offset):
    if(self.config['only_random_tasks']):
      task_type = Tasktype.RANDOM
    else:
      task_type = np.random.choice(self.adaptive_filtering_task_types, 1, p=self.adaptive_filtering_task_probs)[0] # adaptive sampling of the task type
    return self.get_dynamic_task(task_type,translation_offset)

  def get_dynamic_task(self, task_type, translation_offset):
    """
      This is the main algorithm for nav-acl-q
    """

    
    if(self.config['use_AF_database']):
      mu, sigma = self.fast_create_adaptive_boundaries_mu_sig()
    else:
      mu, sigma = self.create_adaptive_boundaries()

    beta      = self.config['adaptive_filtering_params']['nav_beta']
    gamma_low = self.config['adaptive_filtering_params']['nav_gamma_low']
    gamma_hi  = self.config['adaptive_filtering_params']['nav_gamma_hi']
    omega     = self.config['adaptive_filtering_params']['nav_P_omega']


    for trial in range(self.config['nav_acl_max_AF_task_samples']): 
      if(self.config['runtimetasks_from_database'] and self.config['use_AF_database']):
        robot_task = self.sample_task_from_database(self.AF_database)
      else:
        robot_task = self.sample_random_task(translation_offset)
      succ_prob  = self.get_task_success_probability(robot_task).cpu()
      time.sleep(0.01)

      if task_type == Tasktype.EASY:
        if (mu + beta*sigma) < 1:
          if (succ_prob > (mu + beta*sigma)) or (succ_prob > omega):
            robot_task.task_type = Tasktype.EASY
            return robot_task
        else:
          if (succ_prob > mu):
            robot_task.task_type = Tasktype.EASY
            return robot_task
      if task_type == Tasktype.FRONTIER:
        if (mu - gamma_low*sigma) < succ_prob < (mu + gamma_hi*sigma):
          robot_task.task_type = Tasktype.FRONTIER
          return robot_task
      if task_type == Tasktype.RANDOM: 
        robot_task.task_type = Tasktype.RANDOM
        return robot_task
    print("TOOK too many trials to find task of type: ", task_type, " with boundaries mu: ", mu, " sigma: ", sigma)
    robot_task.task_type = Tasktype.RANDOM
    return robot_task


  def create_adaptive_boundaries(self, translation_offset=np.array([0,0,0])):
    tasks_parameters = []
    if(self.config['use_AF_database']):
      tasks_parameters = self.get_task_parameters_for_AF_database()

    else: 
      for i in range(self.config['nav_acl_AF_boundaries_task_samples']):
        tasks_parameters.append(self.get_task_params_array(self.sample_random_task(translation_offset=translation_offset),self.config['normalize_tasks']))

    params      = torch.FloatTensor(tasks_parameters).to(self.device)
    predictions = self.NavACLNet(params).cpu().detach().numpy() 
    mu, sigma   = scipy.stats.distributions.norm.fit(predictions)
    return mu, sigma


  def get_task_parameters_for_AF_database(self):
    tasks_parameters = []
    for i in range(self.config['nav_acl_AF_boundaries_task_samples']):
      task = self.sample_task_from_database(self.AF_database)
      tasks_parameters.append(self.get_task_params_array(task,self.config['normalize_tasks'])) # and append it to the task array which will be used for creating the adaptive boundaries
    return tasks_parameters

  def sample_task_from_database(self,database):
    task_sample                        = random.sample(database, 1)[0]
    obs_cam, obs_lidar,_,_,task_array  = task_sample
    prev_action                        = np.zeros(8)
    prev_reward                        = np.ones(4)*self.config['step_penalty']
    q                                  = self.observation_to_q_value(obs_cam,obs_lidar,prev_action,prev_reward) # pass observation through the q network to obtain currently estimated q value
    task                               = RobotTask(self.default_task)                                           # create a task instance
    task.from_task_array(task_array)                                                                            # initialize according to the task that was sampled from the database
    task.q_value = q                                                                                            # replace the q 
    return task

  def compute_dolly_robot_distance(self, task):
      return distance.euclidean(task.robot_translation, task.dolly_translation)
  
  def compute_obstacle_distance_dolly(self, task):
      distances = np.array([distance.euclidean(task.obstacle_2_translation, task.dolly_translation),
                            distance.euclidean(task.obstacle_3_translation, task.dolly_translation)])
      return np.min(distances)

  def compute_obstacle_distance_robot(self, task):
      distances = np.array([distance.euclidean(task.obstacle_translation,   task.robot_translation),
                            distance.euclidean(task.obstacle_1_translation, task.robot_translation)])
      return np.min(distances)

  def compute_path_complexity(self, task):
      return 0.1

  def check_if_task_is_valid(self,task):
    # first check is no object is too close to the dolly:
    if(self.compute_dolly_min_distance(task) < self.config['randomization_params']['min_dist_dolly_obs']):
      return False
    elif(self.compute_robot_min_distance(task) < self.config['randomization_params']['min_dist_robot_obs']):
      return False
    else:
      return True

  def compute_robot_min_distance(self,task):
    return self.compute_min_distance(task, task.robot_translation)
  def compute_dolly_min_distance(self,task):
    return self.compute_min_distance(task, task.dolly_translation)
  
  def compute_min_distance(self,task, target_translation):
    distances = np.zeros((4))
    obstacles = task.get_obstacle_translations_array()
    for i in range(4):
      distances[i] = distance.euclidean(obstacles[i], target_translation)
    min_dist = np.min(distances)

    if min_dist > 50:
      return 0
    else : 
      return np.min(distances)

  def compute_relative_rotation_robot_dolly(self, task, as_degrees=False):
    Robot_orientation     = rot_vec(self.initial_orientation,task.robot_rotation[0])
    Dolly_orientation     = rot_vec(self.initial_orientation,task.dolly_rotation[0])
    relative_rotation_rad = rel_rot(Robot_orientation,Dolly_orientation)
    if(as_degrees):
      return relative_rotation_rad * (180/np.pi)
    else:
      return relative_rotation_rad



  def train(self,robot_task,label,batch_mode = False):
    self.NavACL_optimizer.zero_grad()

    if(self.config['nav_acl_batch_mode']):
      params   = []
      labels   = []
      for t in range(self.config['nav_acl_batch_size']):
        if(type(robot_task[t])== RobotTask):
          params.append(self.get_task_params_array(robot_task[t],self.config['normalize_tasks']))
        else:
          params.append(self.normalize_task_params_array(robot_task[t]))
        labels.append(label[t])

      task_params_array = torch.FloatTensor(params).to(self.device)
      label             = torch.FloatTensor(labels).flatten().to(self.device)


    else: 
      if(type(robot_task)== RobotTask):
        task_params_array = self.get_task_params_array(robot_task,self.config['normalize_tasks'])
      else:
        task_params_array = self.normalize_task_params_array(robot_task)

      task_params_array = torch.FloatTensor(task_params_array).flatten().to(self.device)
      label             = torch.FloatTensor(label).flatten().to(self.device)


    prediction = self.NavACLNet(task_params_array)
    loss = self.NavACL_loss_func(prediction, label)

    loss.backward()
    self.NavACL_optimizer.step()
    return prediction.detach().cpu().numpy()

  def batch_train(self,robot_tasks, labels):
    self.NavACL_optimizer.zero_grad()

    params   = []
    _labels   = []
    for t in range(self.config['nav_acl_batch_size']):
      if(type(robot_tasks[t])== RobotTask):
        params.append(self.get_task_params_array(robot_tasks[t],self.config['normalize_tasks']))
      else:
        params.append(robot_tasks[t])

      _labels.append(labels[t])


    task_params_array = torch.FloatTensor(params).to(self.device)
    label             = torch.FloatTensor(_labels).flatten().to(self.device)
    prediction        = self.NavACLNet(task_params_array)
    # print("robot_tasks:" , robot_tasks)
    # print("predictions: ", prediction)
    loss              = self.NavACL_loss_func(prediction, label.view((prediction.shape))) # labels are [batchsize] abd predictions are [batchsize,1] thus labels have to be viewed as [8,1]! This is very important, otherwise it will only use one of the parameters for esitmation

    loss.backward()
    self.NavACL_optimizer.step()
    return prediction.detach().cpu().numpy()

  def create_dtb_stack(self):
      af_database = np.load(self.config['AF_database_path']+str(self.agent_index)+".npy", allow_pickle=True)
      af_database = np.array(af_database).reshape(af_database.shape[0],af_database.shape[1])
      dtb_stack = []
      for t_i in range(af_database.shape[0]):
          image_stack = np.array([af_database[t_i][0],af_database[t_i][0],af_database[t_i][0],af_database[t_i][0]]).reshape(12,80,80)
          lidar       = af_database[t_i][1]
          action      = np.zeros((8))
          reward      = np.ones(4)*self.config['step_penalty']
          task_array  = af_database[t_i][4]
          robot_task  = RobotTask(self.config['default_task_unity'])
          robot_task.from_task_array(task_array)
          task_params_array = self.get_task_params_array(robot_task,self.config['normalize_tasks'])
          task_tuple = (image_stack, lidar, action, reward, task_params_array)
          dtb_stack.append(task_tuple)
      return dtb_stack

  def fast_create_adaptive_boundaries_mu_sig(self):
    c, l, a, r, tsk = map(np.stack, zip(*random.sample(self.dtb_stack, self.config['nav_acl_AF_boundaries_task_samples'])))

    cam_tensor          = torch.FloatTensor(c).to(Qricculum_device)
    lidar_tensor        = torch.FloatTensor(l).to(Qricculum_device)
    prev_action         = torch.FloatTensor(a).to(Qricculum_device)
    prev_reward         = torch.FloatTensor(r).to(Qricculum_device)  # prev_reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
    with torch.no_grad():
        predicted_q_values = self.q_net(cam_tensor,lidar_tensor, prev_action, prev_reward).detach().cpu().numpy()
        predicted_q_values = np.interp(predicted_q_values, (self.config['collision_penalty']-self.config['goal_reward'],self.config['goal_reward']), (-1,1))
    
    tsk[:,4] = predicted_q_values[:,0]
    # print(tsk)
    tsk_tensor = torch.FloatTensor(tsk).to(self.device)
    predictions = self.NavACLNet(tsk_tensor).detach().cpu().numpy()
    mu, sigma   = scipy.stats.distributions.norm.fit(predictions)
    return mu, sigma