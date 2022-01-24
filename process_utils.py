from numpy.core.shape_base import stack
from RobotTask import RobotTask
import json

from scipy.signal.ltisys import StateSpace
from VisualTaskEnv import VisualTaskEnv
from isaac import *
import time
import numpy as np
from threading import Thread
from utils import *
import torch
from PyController import PyController
from nav_acl import Nav_ACL
from collections import deque
import copy
from Networks import *


def start_gather_experience(agent_index, shared_elements,config):
    ########## start isaac app #######################################
    app                             = Application("apps/py_sac_nav_acl/py_sac_nav_acl.app.json")
    isaac_thread                    = Thread(target=start_isaac_application,args=(app,agent_index,config,))
    isaac_thread.start()
    ###########environment and alrogithm setup########################
    time.sleep(1)
    env = VisualTaskEnv(app,agent_index, "py_controller_"+str(agent_index),config)
    torch.manual_seed((23345*agent_index))
    np.random.seed((23345*agent_index))
    env.seed((23345*agent_index))
    ##########run episodes'############################################
    # run_test_episodes(env, agent_index,shared_elements,config,)
    run_episodes(env, agent_index,shared_elements,config,)
    app.stop()

def run_test_episodes(env, agent_index,shared_elements, config):
    Full_Path = "/home/developer/Training_results/Visual/10/08/2021/16:05:00/"+"_CHECKPOINT_" + "4_19681"
    Full_navn = "/home/developer/Training_results/Visual/10/08/2021/16:05:00/"+"_nav_acl_network_" + "4_19681"

    No_AE_Path= "/home/developer/Training_results/Visual/10/05/2021/11:19:13/"+"_CHECKPOINT_" + "62_541207"
    No_AE_navn= "/home/developer/Training_results/Visual/10/05/2021/11:19:13/"+"_nav_acl_network_" + "62_541207"

    RND_Path  = "/home/developer/Training_results/Visual/09/30/2021/09:44:53/"+"_CHECKPOINT_" + "22_162675"
    RND_navn  = "/home/developer/Training_results/Visual/09/30/2021/09:44:53/"+"_nav_acl_network_" + "22_162675"
    nav_list = [Full_navn,No_AE_navn,RND_navn]
    check_list = [Full_Path,No_AE_Path,RND_Path]
    
    names_list = ["FULL", "No_AE", "RND"]
    nav_acl_network         = NavACLNetwork(5,config['nav_acl_hidden_dim']).to(train_device)
    for model_index in range(len(names_list)):
        print("load model: ", check_list[model_index])
        checkpoint=torch.load(check_list[model_index])
        print("load navacl model: ", nav_list[model_index])
        nav_check =torch.load(nav_list[model_index])
        nav_acl_network.load_state_dict(nav_check)
    print("hat geklappt")

    for model_index in range(len(names_list)):
        checkpoint=torch.load(check_list[model_index])
        nav_check =torch.load(nav_list[model_index])
        nav_acl_network.load_state_dict(nav_check)
        soft_q_net1 = SoftQNetwork(config).to(train_device)
        soft_q_net1.load_state_dict(checkpoint['soft_q_net1'])

        model_clone                   = PolicyNetwork(config)
        model_clone.load_state_dict(checkpoint['policy_net'])
        model_clone.to(inference_device)

        if( (config['q_ricculum_learning']) or config['create_new_AF_database']):
            nav_acl             = Nav_ACL(nav_acl_network,config,shared_q_net=soft_q_net1,env=env,task_offset=np.array([0,0,0]),agent_index=agent_index)
        else: 
            nav_acl             = Nav_ACL(nav_acl_network,config,agent_index=agent_index)
        for param in model_clone.parameters():
            param.requires_grad = False
        test_grid = generate_test_grid()
        test_angles = [0,45,-45,90,-90]
        # print(test_grid)
        
        for angle in test_angles:
            for trial in range(0,4):
                print("agent: ", agent_index, " test case: ", angle, trial, "model: ", names_list[model_index])
                results = []
                for task_offset in test_grid:
                    orig_offset = task_offset
                    task_config = config['default_test_unity']
                    task_config['robot_pose']['rotation_yaw'][0] = angle
                    task = RobotTask(task_config) # create a task instance from a config dict
                    task_offset[1] += (agent_index*40)
                    task = nav_acl.apply_offset_to_robot(task, task_offset)
                    task = nav_acl.apply_offset_to_dolly(task, [0,(agent_index*40),0])

                    task.set_q_value(nav_acl.get_q_value_for_task(task,task_offset))
                    num_steps, task_result, pred_prob = run_episode(env, agent_index, model_clone, nav_acl,shared_elements, 1, 0, config, test_task=task)
                    results.append([orig_offset,angle,nav_acl.get_task_params_array(task,normalize=False),agent_index,num_steps,task_result, pred_prob])
                results_array = np.asarray(results)
                np.save("/home/developer/Testing_results/"+names_list[model_index]+"_gt_agent_"+str(agent_index)+"_trial_" + str(trial)+"_yaw_"+str(angle), results_array)



def run_episodes(env, agent_index, shared_elements, config):
    max_episodes                  = config['max_episodes']
    num_Agents                    = config['num_Agents']
    num_episodes                  = int(max_episodes / num_Agents)
    steps                         = 0
    robot_grid                    = generate_robot_grid()
    task_offset                   = robot_grid[agent_index]
    model_clone                   = PolicyNetwork(config)

    model_clone.load_state_dict(shared_elements['policy_network'].state_dict())
    model_clone.to(inference_device)

    for param in model_clone.parameters():
        param.requires_grad = False

    if( (config['q_ricculum_learning']) or config['create_new_AF_database']):
        nav_acl             = Nav_ACL(shared_elements['nav_acl_network'],config,shared_q_net=shared_elements['q_network'],env=env,task_offset=task_offset,agent_index=agent_index)
    else: 
        nav_acl             = Nav_ACL(shared_elements['nav_acl_network'],config,agent_index=agent_index)
    
    if( config['create_new_AF_database']):
        nav_acl.save_tasks_for_later(task_offset,config['new_AF_database_size'],"/home/developer/Training_results/Qricculum_Learning/"+str(agent_index))
        print("FINISHED SAVING TASK DATABASE FOR QRICCULUM LEARNING, saved at: ","/home/developer/Training_results/Qricculum_Learning/"+str(agent_index))
        nav_acl.load_AF_database("/home/developer/Training_results/Qricculum_Learning/"+str(agent_index)+".npy")
    

    for episode_index in range(num_episodes):
        if (episode_index % config['update_nav_nets_every_N_episodes']) == 0:
            # Create a new local copy of the latest Policy net before starting the episode
            model_clone.load_state_dict(shared_elements['policy_network'].state_dict())
            for param in model_clone.parameters():
                param.requires_grad = False
            if( config['q_ricculum_learning']):
                nav_acl.update_networks(shared_elements['nav_acl_network'],shared_elements['q_network'],task_offset)
            else: 
                nav_acl.update_networks(shared_elements['nav_acl_network'],None)
        
        num_steps, result = run_episode(env, agent_index, model_clone, nav_acl,shared_elements, episode_index, steps, config)
        steps += num_steps



def run_episode(env, agent_index, policy_net, nav_acl,shared_elements, num_of_episode, steps, config, test_task=None):
    num_stacked_frames      = config['num_stacked_frames']  
    train_start             = config['train_starts']
    max_steps               = config['max_steps']
    num_Agents              = config['num_Agents']
    t_max                   = config['buffer_maxlen']*num_Agents
    robot_grid              = generate_robot_grid()
    task_offset             = robot_grid[agent_index]
    t                       = steps*num_Agents # estimate total number of steps in the replay buffer
    P_Random                = min(nav_acl.config['adaptive_filtering_params']['p_random_max'], t/t_max)
    P_Easy                  = ((1-P_Random)/2)
    P_Frontier              = P_Easy
    action_dim              = (2)
    nav_acl.adaptive_filtering_task_probs = [P_Random,P_Easy,P_Frontier]

    if(test_task is None):
        task = nav_acl.generate_random_task(translation_offset=task_offset)
    else:
        print("TEST TASK")
        task = test_task

    pred_prob_worker        = nav_acl.get_task_success_probability(task).detach().cpu().numpy()

    step_durations          = shared_elements['step_durations']
    episode_rewards         = shared_elements['episode_rewards']
    episode_lengths         = shared_elements['episode_lengths']
    intermediate_buffer     = shared_elements['experience_queue']
    intermediate_buffer_nav_acl = shared_elements['nav_acl_exp_queue']


    obs_cam, obs_lidar    =  env.reset(task)
    env.step([0,0])
    obs_cam, obs_lidar    = env.reset(task) 
    stacked_camera_obsrv  = deque(maxlen=num_stacked_frames)
    stacked_actions       = deque(maxlen=num_stacked_frames)
    stacked_rewards       = deque(maxlen=num_stacked_frames)
    transitions           = deque(maxlen=num_stacked_frames)
    episode_reward  = 0
    ep_steps        = 0
    prev_action     = np.zeros(action_dim)
    prev_reward     = config['step_penalty']
    task_result     = [0]
    episode_exp     = []
    # start with a fresh state
    for i in range(num_stacked_frames):
        stacked_camera_obsrv.append(obs_cam)
        stacked_actions.append(prev_action)
        stacked_rewards.append(prev_reward)
        if(config['use_snail_mode']):
            transitions.append((obs_cam, obs_lidar,prev_action,prev_reward))



    start_episode           = current_milli_time()
    
   
    for step in range(max_steps):
        start = current_milli_time()

        stacked_camera_state = np.asarray(stacked_camera_obsrv).reshape((env.observtion_shape[0]*num_stacked_frames,env.observtion_shape[1],env.observtion_shape[2]))
        stacked_prev_action  = np.asarray(stacked_actions) # [a-4,a-3, a-2, a-1]
        stacked_prev_reward  = np.asarray(stacked_rewards).reshape(num_stacked_frames,1)

        with torch.no_grad():
            if steps >= train_start:
                action = policy_net.get_action(stacked_camera_state,obs_lidar,stacked_prev_action,stacked_prev_reward, deterministic = False, device=inference_device)
            else:      
                action = policy_net.sample_action()

        ### do action and get observation
        

        step_start = current_milli_time()
        next_obs_cam, next_obs_lidar, reward, done, info = env.step(action)

        ### append new action and reward (has to happen before appending to the replay buffer!)
        stacked_actions.append(action) # [a-3,a-2,a-1, a]
        stacked_rewards.append(reward) 

        stacked_action  = np.asarray(stacked_actions)
        stacked_reward  = np.asarray(stacked_rewards).reshape(num_stacked_frames,1)

        ### append to episode exp
        episode_exp.append((obs_cam, obs_lidar, stacked_action, stacked_prev_action, stacked_reward, stacked_prev_reward, next_obs_cam, next_obs_lidar, done))


        stacked_camera_obsrv.append(next_obs_cam) # has to happen after appending to the replay buffer

        obs_cam, obs_lidar = next_obs_cam, next_obs_lidar
        episode_reward += reward
       
        end = current_milli_time()
        step_durations.append(end-start)
        prev_action = action
        prev_reward = reward
        ep_steps += 1

        ### some manual cleanup to prevent numpy from polluting my memory
        del stacked_camera_state, next_obs_cam, next_obs_lidar
        
        if done:
            if info['collision']:
                task_result = [0]
            else:
                task_result = [1]

            break
    time_for_episode = current_milli_time()-start_episode
    avg_time_per_step = int(time_for_episode / ep_steps)
    print('Agent: ', agent_index,  'Finished Episode: ', num_of_episode, ' | Pred prob.: ', round(pred_prob_worker[0],2), ' | was considered: ', task.task_type.name, '| Episode Reward: ', round(episode_reward,2), ' | Num steps: ', ep_steps, '| avg time per step: ',avg_time_per_step, ' | done: ', task_result[0])
    episode_rewards.append(episode_reward)
    episode_lengths.append(ep_steps)

    nav_acl.params.append(nav_acl.get_task_params_array(task,False)) # append the non normalized version of the task since we need that information for our statistics

    # append results to the shared queue
    intermediate_buffer_nav_acl.put((task,task_result,agent_index))
    intermediate_buffer.put(episode_exp)
    return ep_steps, task_result[0], pred_prob_worker

def start_isaac_sim_connector(num_Agents=1, start_port=64000):
  """ creates PyCodelets for receiving and sending messages to/from the isaac simulation environment (either Unity or Omniverse) via specified tcp subscriber and publisher nodes to allow for parallel access to the simulation """
  app = Application("apps/py_sac_nav_acl/py_sim_connector.app.json")
  PUB_PORT_NUM = start_port
  SUB_PORT_NUM = start_port+1000
  for index in range(num_Agents):
    #create application and create node
    agent_suffix = '_'+str(index)
    app_name = "connector" +agent_suffix
    
    connector = app.add(app_name)

    tcp_sub = connector.add(app.registry.isaac.alice.TcpSubscriber)
    tcp_pub = connector.add(app.registry.isaac.alice.TcpPublisher)
    connector.add(app.registry.isaac.alice.TimeSynchronizer)

    tcp_sub.config['port'] = SUB_PORT_NUM     + index
    tcp_sub.config['host'] = 'localhost'
    tcp_pub.config['port'] = PUB_PORT_NUM     + index

    #receive messages from sim and publish via TCP
    app.connect('simulation.interface/output',                             'collision'+agent_suffix,       app_name+'/TcpPublisher',        'collision'+agent_suffix)
    app.connect('simulation.interface/output',                             'bodies'+agent_suffix,          app_name+'/TcpPublisher',        'bodies'+agent_suffix)
    app.connect('simulation.interface/output',                             'rangescan_front'+agent_suffix, app_name+'/TcpPublisher',        'rangescan_front'+agent_suffix)
    app.connect('simulation.interface/output',                             'rangescan_back'+agent_suffix,  app_name+'/TcpPublisher',        'rangescan_back'+agent_suffix)
    app.connect('simulation.interface/output',                             'color'+agent_suffix,           app_name+'/TcpPublisher',        'color'+agent_suffix)
    
    #receive messages from TCP and publish them to the simulation
    print("messages coming from: ", SUB_PORT_NUM     + index, "go to :", 'simulation.interface/input/####'+agent_suffix)
    #send control messages
    app.connect(app_name+'/TcpSubscriber',              'diff_command'+agent_suffix,              'simulation.interface/input',                  'base_command'+agent_suffix)
    app.connect(app_name+'/TcpSubscriber',              'teleport_robot'+agent_suffix,            'simulation.interface/input',                  'teleport_robot' +agent_suffix)
    app.connect(app_name+'/TcpSubscriber',              'teleport_dolly'+agent_suffix,            'simulation.interface/input',                  'teleport_dolly' +agent_suffix)
    app.connect(app_name+'/TcpSubscriber',              'teleport_obstacle'+agent_suffix,         'simulation.interface/input',                  'teleport_obstacle' +agent_suffix)
    app.connect(app_name+'/TcpSubscriber',              'teleport_obstacle'+agent_suffix+"_1",    'simulation.interface/input',                  'teleport_obstacle' +agent_suffix+"_1")
    app.connect(app_name+'/TcpSubscriber',              'teleport_obstacle'+agent_suffix+"_2",    'simulation.interface/input',                  'teleport_obstacle' +agent_suffix+"_2")
    app.connect(app_name+'/TcpSubscriber',              'teleport_obstacle'+agent_suffix+"_3",    'simulation.interface/input',                  'teleport_obstacle' +agent_suffix+"_3")

  app.start()
  while True:
      time.sleep(5)
  app.stop()


def start_isaac_application(app, agent_index, config):
  """ creates PyCodelets for receiving and sending messages to/from the isaac simulation environment (either Unity or Omniverse) """
  Port = config['start_port']
  PUB_PORT_NUM = Port + 1000 
  SUB_PORT_NUM = Port

  #create application and create node
  agent_suffix = '_'+str(agent_index)
  app_name = "py_controller" +agent_suffix
  py_controller_node = app.add(app_name)


  component_node = py_controller_node.add(PyController)
  component_node.config['app_name']                     = app_name
  component_node.config['agent_index']                  = agent_index
  component_node.config['tick_period_seconds']          = config['tick_period_seconds']
  component_node.config['action_duration_ticks']        = config['action_duration_ticks']
  component_node.config['depth_cam_max_distance']       = config['depth_cam_max_distance']
  component_node.config['lidar_max_distance']           = config['lidar_max_distance']
  component_node.config['goal_threshold']               = config['goal_threshold']
  component_node.config['output_resolution']            = config['output_resolution']
  component_node.config['using_depth']                  = config['using_depth']
  component_node.config['using_unity']                  = config['using_unity']
  component_node.config['goal_description']             = config['goal_description']
  component_node.config['omniverse_teleport_dict']      = config['omniverse_teleport_dict']
  component_node.config['scale_dolly']                  = config['scale_dolly']
  contact_monitor_node = py_controller_node.add(app.registry.isaac.navigation.CollisionMonitor, 'CollisionMonitor')

  tcp_sub = py_controller_node.add(app.registry.isaac.alice.TcpSubscriber)
  tcp_pub = py_controller_node.add(app.registry.isaac.alice.TcpPublisher)
    
  py_controller_node.add(app.registry.isaac.alice.TimeSynchronizer)

  tcp_sub.config['port'] = SUB_PORT_NUM     + agent_index
  tcp_sub.config['host'] = 'localhost'
  tcp_pub.config['port'] = PUB_PORT_NUM + agent_index
  print("start isaac app: ", agent_index, " \n SUB_PORT: ", SUB_PORT_NUM, "\n PUB_PORT: ", PUB_PORT_NUM, "\n")
  #receive messages
  app.connect(app_name+'/TcpSubscriber',           'collision'+agent_suffix,            app_name+'/CollisionMonitor',        'collision')
  app.connect(app_name+'/CollisionMonitor',        'report',                            app_name+'/PyCodelet',       'collision'+agent_suffix)
  app.connect(app_name+'/TcpSubscriber',           'collision'+agent_suffix,            app_name+'/PyCodelet',       'collision'+agent_suffix)
  app.connect(app_name+'/TcpSubscriber',           'bodies'+agent_suffix,               app_name+'/PyCodelet',       'bodies'+agent_suffix)
  app.connect(app_name+'/TcpSubscriber',           'rangescan_front'+agent_suffix,      app_name+'/PyCodelet',       'lidar_front'+agent_suffix)
  app.connect(app_name+'/TcpSubscriber',           'rangescan_back'+agent_suffix,       app_name+'/PyCodelet',       'lidar_back'+agent_suffix)
  app.connect(app_name+'/TcpSubscriber',           'color'+agent_suffix,                app_name+'/PyCodelet',       'color'+agent_suffix)

  
  print("STARTED CONTROLLER WITH NAME: ", app_name, "RECEIVING TCP ON ", SUB_PORT_NUM +agent_index, "SENDING TCP ON: ", PUB_PORT_NUM+agent_index)
  #send control messages
  app.connect(app_name+'/PyCodelet',              'diff_command'+agent_suffix,         app_name+'/TcpPublisher',      'diff_command'+agent_suffix)
  app.connect(app_name+'/PyCodelet',              'teleport_robot'+agent_suffix,       app_name+'/TcpPublisher',      'teleport_robot'+agent_suffix)
  app.connect(app_name+'/PyCodelet',              'teleport_dolly'+agent_suffix,       app_name+'/TcpPublisher',      'teleport_dolly'+agent_suffix)
  app.connect(app_name+'/PyCodelet',              'teleport_obstacle'+agent_suffix,    app_name+'/TcpPublisher',      'teleport_obstacle'+agent_suffix)
  app.connect(app_name+'/PyCodelet',              'teleport_obstacle'+agent_suffix + "_1",    app_name+'/TcpPublisher',      'teleport_obstacle'+agent_suffix+ "_1")
  app.connect(app_name+'/PyCodelet',              'teleport_obstacle'+agent_suffix + "_2",    app_name+'/TcpPublisher',      'teleport_obstacle'+agent_suffix+ "_2")
  app.connect(app_name+'/PyCodelet',              'teleport_obstacle'+agent_suffix + "_3",    app_name+'/TcpPublisher',      'teleport_obstacle'+agent_suffix+ "_3")
  app.start()
