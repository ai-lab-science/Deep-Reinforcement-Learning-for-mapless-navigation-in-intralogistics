'''
Soft Actor-Critic version 2
using target Q instead of V net: 2 Q net, 2 target Q net, 1 policy net
add alpha loss compared with version 1
paper: https://arxiv.org/pdf/1812.05905.pdf
'''
import numpy as np
import time
from utils import *
import torch.multiprocessing as mp
from process_utils import *
import copy
from threading import Thread
import scipy


    
def mini_batch_train(agent,save_location, config):
        # training loop

    ## These shared lists and managers are for the multi-processing
    manager = mp.Manager()
    episode_rewards         = manager.list()
    episode_lengths         = manager.list()
    step_durations          = manager.list()
    nav_acl_exp_list        = manager.list()
    experience_queue        = manager.Queue()
    nav_acl_exp_queue       = manager.Queue()
    num_Agents              = config['num_Agents']
    nav_acl_network         = NavACLNetwork(5,config['nav_acl_hidden_dim']).to(train_device)

    if config['load_checkpoint']:
        checkpoint              = torch.load(config['checkpoint_path']+"_nav_acl_network_"+str(config['checkpoint_index']))
        nav_acl_network.load_state_dict(checkpoint)

    # observation is only the current front camera image, but we want to concatenate the four most recent frames to form the state therefore we keep track of the 
    # three previous frames in a deque with a maximum length

    agent.policy_net.share_memory()
    agent.target_soft_q_net1.share_memory()
    num_Agents              = config['num_Agents']
    nav_acl_mu_sig          = manager.Array('d', range(2))
    nav_acl_network.share_memory()

    shared_elements         = {"episode_rewards"    : episode_rewards, 
                               "episode_lengths"    : episode_lengths, 
                               "step_durations"     : step_durations, 
                               "policy_network"     : agent.policy_net,
                               "q_network"          : agent.target_soft_q_net1, 
                               "nav_acl_network"    : nav_acl_network, 
                               "experience_queue"   : experience_queue, 
                               "nav_acl_exp_queue"  : nav_acl_exp_queue,
                               "nav_acl_exp_list"   : nav_acl_exp_list,
                               "nav_acl_mu_sig"     : nav_acl_mu_sig}


    update_shared_policy_net(agent, shared_elements)


    dtb_stack = []
    # TODO: checkpoint loading for nav acl network
    
    # start workers: 
    print("START ", num_Agents, " WORKER TRHEADS ")
    processes               = []
    nav_acl_local_exp_list  = [] # this list is used to store the latest n=nav_acl_batch_size task, task_result pairs to train the nav_acl network with

    # ## for each agent run a fraction of the number of episodes in parallel
    for agent_index in range(num_Agents):

        p1 = mp.Process(target=start_gather_experience,args=(agent_index,shared_elements,config,))
        p1.start()

        processes.append(p1)
        time.sleep(0.2)
    
    # start an save cycle for saving the statistics hourly
    t2 = Thread(target=save_cycle,args=(agent,nav_acl_local_exp_list,shared_elements,save_location,))
    t2.start()

    # start the update cycle that will gather the information from the worker cycles to update both the nav-acl network and the policy and q-value networks
    update_cycle(agent,nav_acl_local_exp_list,shared_elements,config,dtb_stack)


def save_cycle(agent,nav_acl_local_exp_list,shared_elements,save_location):
    """
        saves training statistics and networks every hour
    """
    h_of_train  = 0
    while(True):
        time.sleep(180)
        save_training_statiscics_to_location(shared_elements['episode_lengths'],
                                             shared_elements['step_durations'],
                                             shared_elements['episode_rewards'],
                                             nav_acl_local_exp_list,
                                             agent,
                                             shared_elements['nav_acl_network'],
                                             save_location,
                                             h_of_train)
        time.sleep(3420)
        h_of_train += 1


#TODO: make this easier to read
def update_cycle(agent,nav_acl_local_exp_list,shared_elements,config, dtb_stack):
    """
        this cycle is an infinite loop that checks if new exp for training the q and policy networks can be found in the shared_elements['experience_queue']
        if so it will train the q nets and the policy nets (policy net and target_q_net are shared accross all processes) on this new exp. 
        (remember that the workers replace their copy of the qnet and the policy net on every episode)
        Furthermore this cylce looks out for new nav_acl exp inside the shared_elements['nav_acl_exp_queue'], 
        if so the shared nav_acl Network will be trained - either batch wise or for each incoming exp
    """
    train_start             = config['train_starts']
    batch_size              = config['batch_size']
    use_reward_scaling      = config['use_reward_scaling']
    gamma                   = config['gamma']
    num_Agents              = config['num_Agents']
    soft_tau                = config['tau']
    updates                 = 0 
    steps                   = 0
    nav_acl                 = Nav_ACL(shared_elements['nav_acl_network'], config,worker_instance=False)
    nav_acl_batch_tasks     = []
    nav_acl_batch_results   = []
    saved = False
    while(True):
        start = current_milli_time()
        idle = True

        ### This part gathers the experience (for the SAC part and the nav acl part)
        for q in range(num_Agents):
            if (not shared_elements['experience_queue'].empty()):
                episode = shared_elements['experience_queue'].get()
                for exp in episode: # take exp out of the shared queue and put it into the replay buffer
                    obs_cam, obs_lidar, action, prev_action, reward, prev_reward, next_obs_cam, next_obs_lidar, done = exp

                    # put it into the replay buffer
                    agent.replay_buffer.add(obs_cam, obs_lidar, action, prev_action, reward, prev_reward, next_obs_cam, next_obs_lidar, done)
                    steps += 1
                    agent.steps += 1

                idle=False
            if (not shared_elements['nav_acl_exp_queue'].empty()): # take nav acl exp out of the shared queue and use it to train the shared nav-acl network
                task, result, agent_index = shared_elements['nav_acl_exp_queue'].get()
                task_params_array         = nav_acl.get_task_params_array(task,normalize=False) # not normalizing this bc. only used for plotting
                prediction                = nav_acl.get_task_success_probability(task).detach().cpu()

                if(config['q_ricculum_learning']):
                    mu, sig             = 0.5, 0.5 # q_ricculum lerning needs a q value estimate to create an adaptive boundaries, thus if we use q_ricculum lerning we cannot have this statistic so we replace with 0.5, 0.5
                else:
                    mu, sig             = nav_acl.create_adaptive_boundaries(100)

                nav_acl_local_exp_list.append(((task_params_array,result, prediction, mu, sig, agent_index,task.task_type)))

                if(nav_acl.config['nav_acl_batch_mode']):
                    nav_acl_batch_tasks.append(nav_acl.get_task_params_array(task,normalize=nav_acl.config['normalize_tasks'])) 
                    nav_acl_batch_results.append(result)
                    if(len(nav_acl_batch_tasks) == nav_acl.config['nav_acl_batch_size']):
                        nav_acl_batch_tasks_array = np.asarray(nav_acl_batch_tasks)
                        nav_acl_batch_results_array = np.asarray(nav_acl_batch_results)
                        nav_acl.batch_train(nav_acl_batch_tasks_array, nav_acl_batch_results_array)
                        nav_acl_batch_tasks.clear()
                        nav_acl_batch_results.clear()
                        
                else:   
                    task_params_array   = nav_acl.get_task_params_array(task,normalize=nav_acl.config['normalize_tasks'])
                    nav_acl.train(task, result)
        # # update_start = current_milli_time()

        ## after some Experience has been collected and put into the replay buffer, I perform 4 consecutive training steps
        for x in range(4):
            if((steps > config['train_starts']) and (steps > config['batch_size']) and (steps > config['fill_buffer_with_transitions'])):
                if(updates < int(steps/4)):
                    agent.update(batch_size, reward_scale=2., auto_entropy=True, target_entropy=-1.,use_reward_scaling=use_reward_scaling,gamma=gamma,soft_tau=soft_tau)
                    updates+=1
                    idle=False
        # print("update process cycle time: ", current_milli_time()-start, "of which update time: ", current_milli_time()-update_start)
        if(updates % 500 <= 10):
            print("steps: ", steps, " updates so far: ", updates)
            

        if(idle):
            time.sleep(0.1)




## These three are not really needed since the networks in the shared_elements are the ones that are updated in the update cycle!
def update_shared_q_net(agent, shared_elements):
    for target_param, param in zip(shared_elements['q_network'].parameters(), agent.soft_q_net1.parameters()):
        target_param.data.copy_(param.data)

def update_shared_policy_net(agent,shared_elements):
    for target_param, param in zip(shared_elements['policy_network'].parameters(), agent.policy_net.parameters()):
        target_param.data.copy_(param.data)

def update_shared_nav_net(nav_acl, shared_elements):
    for target_param, param in zip(shared_elements['nav_acl_network'].parameters(), nav_acl.NavACLNet.parameters()):    
        target_param.data.copy_(param.data)

def save_training_statiscics_to_location(episode_lengths,step_durations,episode_rewards,nav_acl_exp_list, agent, nav_acl_network, save_location,h_of_train):
    with mp.Lock():
        nav_acl_params           = []
        nav_acl_dones            = []
        nav_acl_predictions      = []
        nav_acl_mu               = []
        nav_acl_sig              = []
        nav_acl_agent_index      = []
        nav_acl_task_type        = []
        copy_episode_lengths     = []
        copy_step_durations      = []
        copy_episode_rewards     = []
    
        for (task_params,result, prediction, mu, sig, agent_index, task_type) in nav_acl_exp_list:
            nav_acl_params.append(task_params)
            nav_acl_dones.append(result)
            nav_acl_predictions.append(prediction)
            nav_acl_mu.append(mu)
            nav_acl_sig.append(sig)
            nav_acl_agent_index.append(agent_index)
            nav_acl_task_type.append(task_type.value)
        
        for step in step_durations:
            copy_step_durations.append(step)
        
        for episode_reward in episode_rewards:
            copy_episode_rewards.append(episode_reward)
        
        for episode_length in episode_lengths:
            copy_episode_lengths.append(episode_length)

        np.save(save_location+"episode_lengts",     copy_episode_lengths)
        np.save(save_location+"episode_rewards",    copy_episode_rewards)
        np.save(save_location+"step_durations",     copy_step_durations)
        np.save(save_location+"q1_losses",          agent.q1_losses)
        np.save(save_location+"q2_losses",          agent.q2_losses)
        np.save(save_location+"policy_losses",      agent.policy_losses)
        np.save(save_location+"alpha_losses",       agent.alpha_losses)
        np.save(save_location+"alpha",              agent.alpha_list)


        np.save(save_location+"nav_acl_params",     nav_acl_params)
        np.save(save_location+"nav_acl_dones",      nav_acl_dones)
        np.save(save_location+"nav_acl_predictions",nav_acl_predictions)
        np.save(save_location+"nav_acl_mu",         nav_acl_mu)
        np.save(save_location+"nav_acl_sig",        nav_acl_sig)
        np.save(save_location+"nav_acl_agent_index",nav_acl_agent_index)
        np.save(save_location+"nav_acl_task_type",  nav_acl_task_type)
        torch.save(nav_acl_network.state_dict(), save_location+'_nav_acl_network_'+str(h_of_train)+"_"+str(len(episode_rewards)))
        agent.save_checkpoint(save_location,str(h_of_train)+"_"+str(len(episode_rewards)))
        agent.save_model(save_location)
        print("SAVED SUCCESSFULLY!")

def test_batch(env,agent,max_steps,num_tests):
    for eps in range(num_tests):
        state =  env.reset()
        episode_reward = 0
        prev_action = agent.policy_net.sample_action()

        for step in range(max_steps):
            action = agent.policy_net.get_action(state,prev_action, deterministic = False)
            next_state, reward, done, _ = env.step(action)            
            episode_reward += reward
            state=next_state
            prev_action = action
            
            if done:
                break

        print('Episode: ', eps, '| Episode Reward: ', episode_reward)