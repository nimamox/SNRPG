import time, os
import torch
import numpy as np
import random

from DSA_env import DSA_Period
from Agent import RL_Agent
import hickle as hkl 
import random

from config import *

def set_seed(seed):
   torch.manual_seed(seed)
   np.random.seed(seed)
   random.seed(seed)

set_seed(1337)


if SCENARIO in ('DSS', 'SSSD'):
   dim_actions = n_channel * (nPOWS + 1) # The action space size
elif SCENARIO == 'SSSC':
   dim_actions = n_channel
else:
   raise Exception('Invalid Scenario')

dim_states = n_channel + 1  # The sensing result space

batch_size = 300
total_episode = batch_size * 200
epsilon_update_period = batch_size * 1
e_greedy_start = 0.7  # 0.5
e_greedy_end = 1.0
e_increase = (e_greedy_end - e_greedy_start) / 190
epsilon = np.ones(n_su) * e_greedy_start


ISI_external_cache = {}

### Initialize variables

reward_SU = np.zeros((n_su, total_episode))

access_PU = np.zeros((n_channel, total_episode))
fail_PU = np.zeros((n_channel, total_episode))

power_SU = np.zeros((n_su, total_episode))
access_SU = np.zeros((n_su, total_episode))
success_SU = np.zeros((n_su, total_episode))
fail_SU = np.zeros((n_su, total_episode))
access_channel_SU = - np.ones((n_su, total_episode))

power_SU = np.zeros((n_su, total_episode))
dataRate_SU = np.zeros((n_su, total_episode))
dataRate_PU = np.zeros((n_channel, total_episode))

### Loading DSA Environment

env = DSA_Period(n_channel, n_su)

### Initialize the sensor

active_sensor = np.zeros((n_su, n_channel)).astype(np.int32)
initial_sensed_channel = np.random.choice(n_channel, n_su)
for k in range(n_su):
   active_sensor[k, initial_sensed_channel[k]] = 1

### Instantiate Agents
scale_max = {
    0: [0.00038551720920229464, 0.0066595179050588805, 0.00027474271047131434, 0.0002726297874581597],
    1: [0.0014953893134123765, 0.5025908280384527, 0.00026904287086142257, 0.0002666098921720822],
    2: [0.0007029372157025171,0.9760727993647771, 0.0002707082455349926, 0.0002689885512201969],
    3: [0.00027025116831491577, 0.00027619702276382267, 0.008420839450478347, 0.0007813210974455475],
    4: [0.00028426019685187995, 0.00068141830168986, 0.0002791359607607724, 0.0002827171490683201],
    5: [0.0002657011093071127, 0.00026646264526990286, 0.00040543780016436025, 0.05408355764945355]}

Agents_list = []
n_layers = 1

for k in range(n_su):
   Agents_list.append(RL_Agent(k, dim_actions, dim_states, batch_size,
                            reward_decay=0.9,
                            e_greedy= e_greedy_start,
                            lr = learning_rate
                            ))
   Agents_list[k].scale_max = np.array(scale_max[k])

### Simulation
tic = time.time()

init_step = 0

# SUs sense the environment and get the sensing result (contains sensing errors)
observation = env.sense(active_sensor, init_step)


if REGRESSOR not in ['LinReg', 'MLP']:
   for k in range(n_su):
      Agents_list[k].spike_encoder(observation[k,:], step=init_step)

recorded_actions = []

mov_avg_reward = []

for step in range(init_step, total_episode):
   # SU choose action based on observation
   action = np.zeros(n_su).astype(np.int32)
   for k in range(n_su):
      action[k] = Agents_list[k].choose_action(observation[k,:])

   recorded_actions.append(action.copy())
   # SU take action and get the reward
   reward = env.access(action, step)

   # Record values
   reward_SU[:, step] = reward

   access_PU[:, step] = env.channel_state
   fail_PU[:, step] = env.fail_PU

   access_SU[:, step] = env.access_SU
   success_SU[:, step] = env.success_SU
   fail_SU[:, step] = env.fail_SU

   access_channel_SU[:, step] = env.access_channel_SU
   
   power_SU[:, step] = env.power_SU
   
   dataRate_SU[:, step] = env.dataRate_SU
   dataRate_PU[:, step] = env.dataRate_PU    

   # update the PU states
   env.render_PU_state()

   # update the SU sensors
   active_sensor = env.render_sensor(action)

   # SU sense the environment and get the sensing result (contains sensing errors)
   if True:#(step+1) < total_episode:
      observation_ = env.sense(active_sensor, step+1)
      if REGRESSOR not in ['LinReg', 'MLP']:
         for k in range(n_su):
            Agents_list[k].spike_encoder(observation_[k,:], step+1)

   # Store one episode (s, a, r, s')
   for k in range(n_su):
      if REGRESSOR not in ['LinReg', 'MLP']:
         state = step % batch_size
         state_ = (step % batch_size) + 1
      else:
         state = observation[k, :]
         state_ = observation_[k, :]
      Agents_list[k].store_transition(state, action[k], reward[k], state_)

   # Each SU learns their DQN model
   if ((step + 1) % (batch_size) == 0):
      # if step == 299:
      #   break
      # if step == ((4*300)-1):
      #   break
      if RLTYPE == 'PG':
         for k in range(n_su):
            if REGRESSOR in ['LinReg', 'MLP']:
               Agents_list[k].learn_PG_conventional()
            elif REGRESSOR in ['SNN', 'SNN_scaled', 'LSM', 'SurrGrad']:
               Agents_list[k].learn_PG_snn(step)
            else:
               raise Exception('Invalid regressor')            
      elif RLTYPE == 'DQN':
         #print("Losses:[", end='')
         for k in range(n_su):
            if REGRESSOR in ['LinReg', 'MLP']:
               Agents_list[k].learn_conventional(batch_size, step, 'normal')
            elif REGRESSOR in ['SNN', 'SNN_scaled', 'LSM', 'SurrGrad']:
               ll = Agents_list[k].learn_snn(step, training_batch_size = tbs, 
                                              training_iteration = ti, replace_target_iter = rti)
               #print('%.4f' % ll, end='\t')
            else:
               raise Exception('Invalid regressor')
            Agents_list[k].epsilon = epsilon[k]
            if (epsilon[k] >= 0.8):
               Agents_list[k].update_lr(0.01)              
         #print("]")
      else:
         raise Exception('Invalid RLTYPE')

   if ((step+1) % (1*batch_size) == 0):
      index = np.arange(step+1-batch_size, step+1)
      print('Training time = %d' % (step + 1))
      print('SU: success = %d;  fail = %d;  access = %d' %
              (np.sum(success_SU[:, index]), np.sum(fail_SU[:, index]), np.sum(access_SU[:, index])))
      print('PU: fail = %d;  access = %d' %
              (np.sum(fail_PU[:, index]), np.sum(access_PU[:, index])))
      print('total_reward = %.4f' % (np.sum(reward_SU[:, index])/batch_size))
      mov_avg_reward.append(np.sum(reward_SU[:, index])/batch_size)
      print("MOVAVGREW: %.4f" % np.mean(mov_avg_reward))

      elapsed = time.time() - tic
      print('Elapsed time = %.4f sec' % elapsed)
      print('-'*20)
 
   if RLTYPE == 'PG':
      pass
   elif RLTYPE == 'DQN':
      # Update epsilon
      if ((step + 1) % epsilon_update_period == 0):
         print('Epsilon:', min(1, epsilon[k] + e_increase))
         for k in range(n_su):
            #epsilon[k] = min(1, epsilon[k] + 0.1)
            epsilon[k] = min(1, epsilon[k] + e_increase)
            #print('SU %d epsilon update to %.3f' % (k+1, epsilon[k]))

   # swap observation
   observation = observation_

elapsed = time.time() - tic


losses = np.stack([Agents_list[k].cost_his for k in range(n_su)])

result = {'reward_SU': reward_SU,
          'access_PU': access_PU,
          'fail_PU': fail_PU,
          'access_SU': access_SU,
          'success_SU': success_SU,
          'fail_SU': fail_SU,
          'access_channel_SU': access_channel_SU,
          'dataRate_SU': dataRate_SU,
          'dataRate_PU': dataRate_PU,
          'power_SU': power_SU,
          'POWS': POWS,
          'nPOWS': nPOWS,
          'elapsed': elapsed,
          'device': device.type,
          'losses': losses,
          'conf': conf,
          'SCENARIO': SCENARIO,
          'REW_TYPE': REW_TYPE,
          'recorded_actions': np.array(recorded_actions)
          }

hkl.dump(result, os.path.join(RESULT_PATH, '{}_PU{}_SU{}.hkl'.format(FNAME, n_channel, n_su)))
