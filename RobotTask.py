from utils import rot_vec
from numpy import savetxt
import numpy as np
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R
import random
from enum import Enum

class Tasktype(Enum):
  RANDOM    = 0
  EASY      = 1
  FRONTIER  = 2

class RobotTask():
  def __init__(self,default_task,):
    super(RobotTask, self).__init__()

    self.robot_translation      = np.array(default_task['robot_pose']['translation']).astype(np.float)
    self.robot_rotation         = np.array(default_task['robot_pose']['rotation_yaw']).astype(np.float)

    self.dolly_translation      = np.array(default_task['dolly_pose']['translation']).astype(np.float)
    self.dolly_rotation         = np.array(default_task['dolly_pose']['rotation_yaw']).astype(np.float)

    self.obstacle_translation   = np.array(default_task['obstacle_pose']['translation']).astype(np.float)
    self.obstacle_rotation      = np.array(default_task['obstacle_pose']['rotation_yaw']).astype(np.float)

    self.obstacle_1_translation = np.array(default_task['obstacle_1_pose']['translation']).astype(np.float)
    self.obstacle_1_rotation    = np.array(default_task['obstacle_1_pose']['rotation_yaw']).astype(np.float)

    self.obstacle_2_translation = np.array(default_task['obstacle_2_pose']['translation']).astype(np.float)
    self.obstacle_2_rotation    = np.array(default_task['obstacle_2_pose']['rotation_yaw']).astype(np.float)

    self.obstacle_3_translation = np.array(default_task['obstacle_3_pose']['translation']).astype(np.float)
    self.obstacle_3_rotation    = np.array(default_task['obstacle_3_pose']['rotation_yaw']).astype(np.float)

    self.q_value                = 0
    self.task_type              = Tasktype.RANDOM


  def randomize_task(self, randomization_params):
    """
      randomizes the current task settings according to the randomization parameters (as a json)
    """
    num_obstacles                = random.randint(0,randomization_params['num_obstacles'])
    del_obstacle_translation     = np.asarray(randomization_params['del_obstacle_translation'])
    self.robot_rotation         += ( (-0.5 + np.random.uniform(0,1)) *np.array(randomization_params['robot_randomization']['rotation_rnd'])           )
    self.dolly_rotation         += ( (-0.5 + np.random.uniform(0,1)) *np.array(randomization_params['dolly_randomization']['rotation_rnd'])           )
    self.obstacle_rotation      += ( (-0.5 + np.random.uniform(0,1)) *np.array(randomization_params['obstacle_randomization']['rotation_rnd'])        )
    self.obstacle_1_rotation    += ( (-0.5 + np.random.uniform(0,1)) *np.array(randomization_params['obstacle_randomization']['rotation_rnd'])        )
    self.obstacle_2_rotation    += ( (-0.5 + np.random.uniform(0,1)) *np.array(randomization_params['obstacle_randomization']['rotation_rnd'])        )
    self.obstacle_3_rotation    += ( (-0.5 + np.random.uniform(0,1)) *np.array(randomization_params['obstacle_randomization']['rotation_rnd'])        )



    self.robot_translation      += ( (-0.5 + np.random.uniform(0,1)) *np.array(randomization_params['robot_randomization']['translation_rnd_xyz'])    )
    #self.dolly_translation      += ( (-0.5 + np.random.uniform(0,1)) *np.array(randomization_params['dolly_randomization']['translation_rnd_xyz'])    )

    min_dist_d_r                = randomization_params['min_dist_dolly_robot']
    max_dist_d_r                = randomization_params['max_dist_dolly_robot']
    dolly_pos_spline            = randomization_params['dolly_pos_spline_angle']

    dolly_distance              = min_dist_d_r + np.random.uniform(0,1)*(max_dist_d_r-min_dist_d_r)        #create a distance vector with min, max distance according to the config file
    dolly_vector                = np.array([1,0])*dolly_distance                                    #create the final dolly position as the robot position+distance vector
    dolly_pos_rot               = ( (-0.5 + np.random.uniform(0,1))) * dolly_pos_spline                    #rotate it around the origin
    dolly_vector                = self.robot_translation + np.hstack((rot_vec(dolly_vector,dolly_pos_rot),np.zeros(1)))

    self.dolly_translation      = dolly_vector


    
    min_dist_r_o                = randomization_params['min_dist_robot_obs']
    max_dist_r_o                = randomization_params['max_dist_robot_obs']

    min_dist_d_o                = randomization_params['min_dist_dolly_obs']
    max_dist_d_o                = randomization_params['max_dist_dolly_obs']
    obs_mean                    = randomization_params['obstacle_pos_spline_mean']
    obstacle_vectors            = []

    for i in range(2):
      obstacle_distance = min_dist_r_o + np.random.uniform(0,1)*(max_dist_r_o-min_dist_r_o)        #create a distance vector with min, max distance according to the config file
      obstacle_pos_rot  = 90 + np.random.uniform(-obs_mean/2,obs_mean/2)*90+(i*180)                                             #rotate it around the origin
      obstacle_vector   = np.array([1,0])*obstacle_distance                                 #create the final object position as the robot position+distance vector


      obstacle_vector   = self.robot_translation + np.hstack((rot_vec(obstacle_vector,obstacle_pos_rot),np.zeros(1)))
      obstacle_vectors.append(obstacle_vector)

    for i in range(2):
      obstacle_distance = min_dist_d_o + np.random.uniform(0,1)*(max_dist_d_o-min_dist_d_o)        #create a distance vector with min, max distance according to the config file
      obstacle_pos_rot  = 90 + np.random.uniform(-obs_mean/2,obs_mean/2)*90+(i*180)                                           #rotate it around the origin
      obstacle_vector   = np.array([1,0])*obstacle_distance                                 #create the final object position as the dolly position+distance vector


      obstacle_vector   = self.dolly_translation + np.hstack((rot_vec(obstacle_vector,obstacle_pos_rot),np.zeros(1)))
      obstacle_vectors.append(obstacle_vector)

    for remove_obj_idx in range(4-1, num_obstacles-1, -1):
      obstacle_vectors[remove_obj_idx] =  del_obstacle_translation

    # now randomly assign the created obstacle vectors to the target translations
    rnd_obstcl_indexes = np.random.permutation(np.array([0,1,2,3]))
    self.obstacle_translation   =obstacle_vectors[rnd_obstcl_indexes[0]]        #+= ( (-0.5 + np.random.uniform(0,1)) *np.array(randomization_params['obstacle_randomization']['translation_rnd_xyz']) )
    self.obstacle_1_translation =obstacle_vectors[rnd_obstcl_indexes[1]]        #+= ( (-0.5 + np.random.uniform(0,1)) *np.array(randomization_params['obstacle_randomization']['translation_rnd_xyz']) )
    self.obstacle_2_translation =obstacle_vectors[rnd_obstcl_indexes[2]]        #+= ( (-0.5 + np.random.uniform(0,1)) *np.array(randomization_params['obstacle_randomization']['translation_rnd_xyz']) )
    self.obstacle_3_translation =obstacle_vectors[rnd_obstcl_indexes[3]]        #+= ( (-0.5 + np.random.uniform(0,1)) *np.array(randomization_params['obstacle_randomization']['translation_rnd_xyz']) )


  def set_q_value(self, q_value):
    self.q_value = q_value

  def get_obstacle_translations_array(self):
    return np.array([self.obstacle_translation, self.obstacle_1_translation, self.obstacle_2_translation, self.obstacle_3_translation])

  def get_task_array(self):
      """
        returns the task as a numpy array in the form: [[robo_trans_x,robo_trans_y,robo_trans_z,robo_yaw_angle],
                                                        [dolly_trans_x,dolly_trans_y,dolly_trans_z,dolly_yaw_angle],
                                                        [obstacle_trans_x,obstacletrans_y,obstacletrans_z,obobstacleyaw_angle]
                                                        [obstacle_1_trans_x,obstacle_1_trans_y,obstacle_1_trans_z,obobstacle_1_yaw_angle]
                                                        [obstacle_2_trans_x,obstacle_2_trans_y,obstacle_2_trans_z,obobstacle_2_yaw_angle]
                                                        [obstacle_3_trans_x,obstacle_3_trans_y,obstacle_3_trans_z,obobstacle_3_yaw_angle]]
      """
      robot_part      = np.hstack((self.robot_translation,      self.robot_rotation))
      dolly_part      = np.hstack((self.dolly_translation,      self.dolly_rotation))
      obstacle_part   = np.hstack((self.obstacle_translation,   self.obstacle_rotation))
      obstacle_1_part = np.hstack((self.obstacle_1_translation, self.obstacle_1_rotation))
      obstacle_2_part = np.hstack((self.obstacle_2_translation, self.obstacle_2_rotation))
      obstacle_3_part = np.hstack((self.obstacle_3_translation, self.obstacle_3_rotation))

      task_array = np.array([robot_part,dolly_part,obstacle_part,obstacle_1_part,obstacle_2_part,obstacle_3_part])
      return task_array
  
  def from_task_array(self,task_array):
    self.robot_translation      = task_array[0,0:3]
    self.robot_rotation         = np.array([task_array[0][3]])
    self.dolly_translation      = task_array[1,0:3]
    self.dolly_rotation         = np.array([task_array[1][3]])
    self.obstacle_translation   = task_array[2,0:3]
    self.obstacle_rotation      = np.array([task_array[2][3]])
    self.obstacle_1_translation = task_array[3,0:3]
    self.obstacle_1_rotation    = np.array([task_array[3][3]])
    self.obstacle_2_translation = task_array[4,0:3]
    self.obstacle_2_rotation    = np.array([task_array[4][3]])
    self.obstacle_3_translation = task_array[5,0:3]
    self.obstacle_3_rotation    = np.array([task_array[5][3]])
    self.q_value                = 0
  
  


