from config import *
import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import scipy.signal as signal
import time

from torch import nn
import torch.nn.functional as F

import NeuralEncBenchmark
from NeuralEncBenchmark.ttfs import TTFS_encoder
from NeuralEncBenchmark.isi import ISI_encoding
from NeuralEncBenchmark.multiplexing_ttfs import multiplexing_encoding_TTFS_phase
from NeuralEncBenchmark.multiplexing_isi import multiplexing_encoding_ISI_phase
from NeuralEncBenchmark.datasets import *

from NeuralEncBenchmark.torch_device import dtype
from NeuralEncBenchmark.sparse_data_generator import sparse_generator
from NeuralEncBenchmark.surrogate_encoder import encode_data

from NeuralEncBenchmark.surrogate_model import run_snn
from NeuralEncBenchmark.surrogate_train import init_model, compute_classification_accuracy, train

import sys
sys.path.append('PCRITICAL')

from modules.pcritical import PCritical
from modules.utils import OneToNLayer
from modules.topologies import SmallWorldTopology

from Models import *

ISI_external_cache = {}

class RL_Agent():
   def __init__(self, 
                agent_id,
                n_actions,
                n_features,
                memory_size,
                reward_decay=0.9,
                e_greedy=0.9,
                lr=0.01
                ):
      self.agent_id = agent_id
      self.n_actions = n_actions
      self.n_features = n_features
      #self.n_layers = n_layers
      self.gamma = reward_decay
      self.memory_size = memory_size
      self.batch_size = memory_size
      self.epsilon = e_greedy 
      self.scale_max = None

      # total learning step
      self.learn_step_counter = 0

      # initialize learning rate
      self.lr = lr

      # build net
      self.criterion = None
      self.optimizer = None
      
      if REGRESSOR in ['SurrGrad', 'SNN']:
         if CONV_TYPE == 1:
            self.convert_state_scaled = self.convert_state_scaled_1
         elif CONV_TYPE == 2:
            self.convert_state_scaled = self.convert_state_scaled_2
         elif CONV_TYPE == 3:
            self.convert_state_scaled = self.convert_state_scaled_3
         else:
            raise Exception('Invalid conversion')

      if RLTYPE == 'DQN':
         #Criterion
         self.criterion = nn.MSELoss() 
         # initialize zero memory [s, a, r, s_]
         if REGRESSOR not in ['LinReg', 'MLP']:
            self.memory = np.zeros((self.memory_size, 2 + 2))
         else:
            self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
         self.whole_memory = []
         # Make models
         self._build_net_QLEARNING()
      elif RLTYPE == 'PG':
         self.pg_observations = []
         self.pg_actions = []
         self.pg_rewards = []
         self.softmax_fn = nn.Softmax(dim=1)
         self._build_net_PG()
      else:
         raise Exception('Invalid RL mode')
      self.cost_his = []


   def _build_net_PG(self):
      if self.agent_id == 0:
         print("Policy Gradient")
         print('Creating Regressors:', REGRESSOR)   

      if USE_LSM:
         topology = SmallWorldTopology(
            SmallWorldTopology.Configuration(
               minicolumn_shape=minicol,
               macrocolumn_shape=macrocol,
                  p_max=PMAX,
                  # minicolumn_spacing=1460,
                  # intracolumnar_sparseness=635.0,
                  # neuron_spacing=40.0,
                  spectral_radius_norm=SpecRAD,
                  inhibitory_init_weight_range=(0.1, 0.3),
                  excitatory_init_weight_range=(0.2, 0.5),
            )
         )
         lsm_N = topology.number_of_nodes()
         N_inputs = 5
         if CONV_TYPE == 3:
            N_inputs = 6
         self.reservoir = PCritical(1, topology, alpha=ALPHA).to(device)
         #self.lsm = torch.nn.Sequential(OneToNLayer(1, N_inputs, lsm_N), self.reservoir).to(device)
         self.lsm = torch.nn.Sequential(InputLayer(1, N_inputs, lsm_N),
                                        self.reservoir,
                                        ReadoutLayer(lsm_N, readout_inp, readout_out)
                                        ).to(device)

      if REGRESSOR == 'LinReg':
         self.policy_net =  LinReg(self.n_features, self.n_actions)
         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
         
      elif REGRESSOR == 'MLP':
         self.policy_net =  MLP(self.n_features, self.n_actions, hidden)
         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
         #if SCENARIO == 'SSSC':
            #self.power_nets = []
            #self.power_nets_opts = []
            
         
      elif REGRESSOR == 'SurrGrad':
         self.snn_params = {}
         if USE_LSM:
            self.snn_params['dim_in'] = readout_out
         else:
            self.snn_params['dim_in'] = 5
            if CONV_TYPE == 3:
               self.snn_params['dim_in'] = 6

         self.snn_params['T_sim'] = 10
         self.policy_net, self.surr_alpha, self.surr_beta = init_model(self.snn_params['dim_in'], hidden, self.n_actions, .05)
         self.optimizer = optim.Adam(self.policy_net, lr=self.lr, betas=(0.9, 0.999)) #TODO: learning rate
         self.all_obs_spikes = []
         
      elif REGRESSOR.startswith('SNN'):
         self.snn_params = {
            'seed': 1337,
            'Rd': 5.0e3,    # this device resistance is mannually set for smaller leaky current?
            'Cm': 3.0e-6,   # real capacitance is absolutely larger than this value
            'Rs': 1.0,      # this series resistance value is mannually set for larger inject current?

            'Vth': 0.8,     # this is the real device threshould voltage
            'V_reset': 0.0, 

            'dt': 1.0e-6,   # every time step is dt, in the one-order differential equation of neuron
            'T_sim': 10,   # could control total spike number collected
            'dim_in': 5,
            'dim_h': hidden,
            'dim_out': self.n_actions,
            'epoch': 10,

            'W_std1': 1.0,
            'W_std2': 1.0,
         }

         if USE_LSM:
            self.snn_params['dim_in'] = readout_out
         else:
            self.snn_params['dim_in'] = 5
            if CONV_TYPE == 3:
               self.snn_params['dim_in'] = 6

         self.policy_net =  Three_Layer_SNN(self.snn_params)
         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
         self.all_obs_spikes = []
      else:
         raise Exception('Invalid regressor')


   def _build_net_QLEARNING(self):
      if self.agent_id == 0:
         print("Q-Learning")
         print('Creating Regressors:', REGRESSOR)       

      if USE_LSM:
         topology = SmallWorldTopology(
            SmallWorldTopology.Configuration(
               minicolumn_shape=minicol,
               macrocolumn_shape=macrocol,
                  p_max=PMAX,
                  # minicolumn_spacing=1460,
                  # intracolumnar_sparseness=635.0,
                  # neuron_spacing=40.0,
                  spectral_radius_norm=SpecRAD,
                  inhibitory_init_weight_range=(0.1, 0.3),
                  excitatory_init_weight_range=(0.2, 0.5),
            )
         )
         lsm_N = topology.number_of_nodes()
         N_inputs = 5
         if CONV_TYPE == 3:
            N_inputs = 6
         self.reservoir = PCritical(1, topology, alpha=ALPHA).to(device)
         #self.lsm = torch.nn.Sequential(OneToNLayer(1, N_inputs, lsm_N), self.reservoir).to(device)
         self.lsm = torch.nn.Sequential(InputLayer(1, N_inputs, lsm_N),
                                        self.reservoir,
                                        ReadoutLayer(lsm_N, readout_inp, readout_out)
                                        ).to(device)

      if REGRESSOR == 'LinReg':
         self.eval_net =  LinReg(self.n_features, self.n_actions)
         self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
         self.target_net = LinReg(self.n_features, self.n_actions)
         self.target_net.load_state_dict(self.eval_net.state_dict())
         self.target_net.eval()

      elif REGRESSOR == 'MLP':
         hid = 20
         self.eval_net =  MLP(self.n_features, self.n_actions, hid)
         self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
         self.target_net = MLP(self.n_features, self.n_actions, hid)
         self.target_net.load_state_dict(self.eval_net.state_dict())
         self.target_net.eval()

      elif REGRESSOR == 'SurrGrad':
         self.snn_params = {}
         if USE_LSM:
            self.snn_params['dim_in'] = readout_out
         else:
            self.snn_params['dim_in'] = 5
            if CONV_TYPE == 3:
               self.snn_params['dim_in'] = 6

         self.snn_params['T_sim'] = 10
         self.eval_net, self.surr_alpha, self.surr_beta = init_model(self.snn_params['dim_in'], hidden, self.n_actions, .05)
         self.target_net = []
         for vv in self.eval_net:
            self.target_net.append(vv.clone())
         self.optimizer = optim.Adam(self.eval_net, lr=self.lr, betas=(0.9, 0.999)) #TODO: learning rate
         self.all_obs_spikes = []

      elif REGRESSOR.startswith('SNN'):
         self.snn_params = {
            'seed': 1337,
            'Rd': 5.0e3,    # this device resistance is mannually set for smaller leaky current?
            'Cm': 3.0e-6,   # real capacitance is absolutely larger than this value
            'Rs': 1.0,      # this series resistance value is mannually set for larger inject current?

            'Vth': 0.8,     # this is the real device threshould voltage
            'V_reset': 0.0, 

            'dt': 1.0e-6,   # every time step is dt, in the one-order differential equation of neuron
            'T_sim': 10,   # could control total spike number collected
            'dim_in': 5,
            'dim_h': hidden,
            'dim_out': self.n_actions,
            'epoch': 10,

            'W_std1': 1.0,
            'W_std2': 1.0,
         }

         if USE_LSM:
            self.snn_params['dim_in'] = readout_out
         else:
            self.snn_params['dim_in'] = 5
            if CONV_TYPE == 3:
               self.snn_params['dim_in'] = 6

         self.eval_net =  Three_Layer_SNN(self.snn_params)
         self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.lr)
         self.target_net =  Three_Layer_SNN(self.snn_params)
         self.target_net.load_state_dict(self.eval_net.state_dict())
         self.target_net.eval()
         self.all_obs_spikes = []
      else:
         raise Exception('Invalid regressor')

   def store_transition(self, s, a, r, s_):
      if RLTYPE == 'PG':
         self.pg_observations.append(s)
         self.pg_actions.append(a)
         self.pg_rewards.append(r)
      elif RLTYPE == 'DQN':
         if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

         transition = np.hstack((s, [a, r], s_))

         # replace the old memory with new memory
         index = self.memory_counter % self.memory_size
         self.memory[index, :] = transition   
         self.memory_counter += 1    

   def spike_encoder(self, observation, step=None):
      if USE_LSM:
         observation = observation[np.newaxis, :]
         observation = self.convert_state_scaled(observation)
         obs_pois = self.SEncoding(observation)
         obs_spikes = []
         for t in range(self.snn_params['T_sim']):
            S = self.lsm(obs_pois[:,:,t])
            obs_spikes.append(S)
         self.obs_spikes = torch.stack(obs_spikes, dim=2)
         obs_spikes_reshaped = self.obs_spikes.detach().reshape(self.obs_spikes.shape[1], 
                                                                self.obs_spikes.shape[2])
         if REGRESSOR == 'SurrGrad':
            self.obs_spikes = torch.einsum('ijk->ikj', self.obs_spikes)
            self.all_obs_spikes.append(torch.einsum('ij->ji', obs_spikes_reshaped))
         else:
            self.all_obs_spikes.append(obs_spikes_reshaped)
      else:
         observation = observation[np.newaxis, :]
         observation = self.convert_state_scaled(observation)
         self.obs_spikes = self.SEncoding(observation)
         obs_spikes_reshaped = self.obs_spikes.detach().reshape(self.obs_spikes.shape[1], 
                                                                self.obs_spikes.shape[2])
         if REGRESSOR == 'SurrGrad':
            self.obs_spikes = torch.einsum('ijk->ikj', self.obs_spikes)
            self.all_obs_spikes.append(torch.einsum('ij->ji', obs_spikes_reshaped))
         else:
            self.all_obs_spikes.append(obs_spikes_reshaped)

   def choose_action(self, observation):
      if RLTYPE == 'DQN':
         return self.choose_action_DQN(observation)
      elif RLTYPE == 'PG':
         return self.choose_action_PG(observation)

   def choose_action_PG(self, observation):
      observation = observation[np.newaxis, :]
      if REGRESSOR in ['LinReg', 'MLP']:
         logit = self.policy_net(torch.Tensor(observation))
         action = Categorical(logits=logit).sample()[0].detach()
         return action
      elif REGRESSOR == 'SurrGrad':
         with torch.no_grad():
            logit = self.run_surr_grad_snn(self.policy_net, self.obs_spikes)
            action = Categorical(logits=logit).sample()[0].detach()
      elif REGRESSOR.startswith('SNN'):
         with torch.no_grad():
            logit = self.run_ncomm_snn(self.policy_net, self.obs_spikes)
            action = Categorical(logits=logit).sample()[0].detach()
      else:
         raise Exception('Invalid Regressor')
      return action

   def choose_action_DQN(self, observation):
      # to have batch dimension
      observation = observation[np.newaxis, :]

      if np.random.uniform() < self.epsilon:
         # forward feed the observation and get q value for every actions
         if REGRESSOR in ['LinReg', 'MLP']:
            actions_value = self.eval_net(torch.Tensor(observation))
            action = actions_value.max(1)[1][0].detach()
         elif REGRESSOR == 'SurrGrad':
            with torch.no_grad():
               result = self.run_surr_grad_snn(self.eval_net, self.obs_spikes)
               action = result.max(1)[1][0].detach()
         elif REGRESSOR.startswith('SNN'):
            with torch.no_grad():
               self.eval_net.eval()
               result = self.run_ncomm_snn(self.eval_net, self.obs_spikes)
               action = result.max(1)[1][0].detach()
         else:
            observation = self.convert_state_scaled(observation)
            raise Exception('Invalid regressor')
         #actions_value = self.eval_net.predict(observation)
         #actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
         #action = np.argmax(actions_value)
      else:
         action = np.random.randint(0, self.n_actions)
      return action

   def convert_state_scaled_1(self, observation):
      normalizer_max = 2.06
      new_obs = observation.copy()
      new_obs[:,0] = new_obs[:,0] / normalizer_max
      new_obs[:,1:] = new_obs[:,1:] 
      return new_obs

   def convert_state_scaled_2(self, observation):
      normalizer_max = 2.06
      ch = np.argmax(observation[:,1:], 1)
      new_obs = np.zeros([observation.shape[0], observation.shape[1]])
      # new_obs[np.arange(new_obs.shape[0]), ch+1] = observation[:,0] / self.scale_max[ch]
      new_obs[np.arange(new_obs.shape[0]), ch+1] = observation[:,0] / (self.scale_max[ch])
      new_obs[:,0] = observation[:,0] / normalizer_max
      return new_obs

   def convert_state_scaled_3(self, observation):
      normalizer_max = 2.06
      ch = np.argmax(observation[:,1:], 1)
      new_obs = np.zeros([observation.shape[0], observation.shape[1]+1])
      new_obs[:,0] = observation[:,0] / normalizer_max
      new_obs[:,1] = observation[:,0] / (self.scale_max[ch])
      new_obs[:,2:] = observation[:,1:] 
      return new_obs

   def SEncoding(self, X):
      if ENCODER == 'Poisson':
         return Poisson_encoder(torch.Tensor(X), self.snn_params['T_sim'])
      elif ENCODER == 'ISI':
         ed = encode_data(X, X, nb_units=X.shape[1], encoder_type="ISI_inverse", batch_size=X.shape[0], nb_steps=10, TMAX=10, external_ISI_cache=ISI_external_cache)
         ft = next(sparse_generator(ed, shuffle=False))[0]
         return torch.einsum('ijk->ikj', ft.to_dense())
      elif ENCODER == 'Phase+ISI':
         ed = encode_data(X, X, nb_units=X.shape[1], encoder_type="Phase+ISI_inverse", batch_size=X.shape[0], nb_steps=10, TMAX=10, external_ISI_cache=ISI_external_cache, smo_freq=500)
         ft = next(sparse_generator(ed, shuffle=False))[0]
         return torch.einsum('ijk->ikj', ft.to_dense())         
      else:
         raise Exception('Invalid Encoding')

class RL_Agent(RL_Agent):
   def run_ncomm_snn(self, network, inp_spike):
      network.reset_(inp_spike.shape[0])
      out_vs = []
      # inp_spike = Poisson_encoder(torch.Tensor(X), self.snn_params['T_sim'])
      for t in range(self.snn_params['T_sim']):
         out_spike, out_v = network(inp_spike[:,:,t])
         out_vs.append(out_v)
      out_vs = torch.stack(out_vs, dim=2)
      return out_vs.sum(dim=2)

   def run_surr_grad_snn(self, network, inp_spike):
      surr_out, _ = run_snn(inp_spike, inp_spike.shape[0], 
                            self.snn_params['T_sim'], network, self.surr_alpha, self.surr_beta)
      return surr_out.sum(1)

   def _pg_discount_norm_rewards(self):
      # Slow version
      #discounted_rewards2 = np.zeros_like(self.pg_rewards)
      #running_add = 0
      #for t in reversed(range(len(self.pg_rewards))):
         #running_add = running_add * self.gamma + self.pg_rewards[t]
         #discounted_rewards2[t] = running_add
      #print('A', discounted_rewards2[-5:])
      # Faster vectorized version
      discounted_rewards = signal.lfilter([1], [1, -self.gamma], self.pg_rewards[::-1])[::-1]
      #print('B', discounted_rewards[-5:])
      # Normalize episode rewards
      discounted_rewards -= np.mean(discounted_rewards)
      discounted_rewards /= (np.std(discounted_rewards) + 1e-9)
      return discounted_rewards


   def learn_PG_conventional(self):
      discounted_pg_rewards_norm = self._pg_discount_norm_rewards()

      self.pg_observations = np.vstack(self.pg_observations)

      for ep in range(pgepochs):
         self.optimizer.zero_grad()
         logits = self.policy_net(torch.Tensor(self.pg_observations))
         # Old PG loss (identical)
         #logp = Categorical(logits=logits).log_prob(torch.FloatTensor(self.pg_actions))
         #loss = -(logp * torch.FloatTensor(self.pg_rewards)).mean()
         #loss = -(logp * torch.FloatTensor(discounted_pg_rewards_norm.copy())).mean()
         
         # PG loss
         cross_ent = F.nll_loss(F.log_softmax(logits), 
                                target=torch.LongTensor(self.pg_actions), 
                                reduction="none")
         pg_loss = torch.sum(cross_ent * torch.FloatTensor(discounted_pg_rewards_norm.copy()))
         
         # Entropy loss
         if ent_coef:
            policy = F.softmax(logits, dim=-1)
            log_policy = F.log_softmax(logits, dim=-1)
            entropy_loss = torch.sum(policy * log_policy)
            total_loss = pg_loss + ent_coef * entropy_loss
         else:
            total_loss = pg_loss
         
         total_loss.backward()
         self.optimizer.step()

      self.pg_actions, self.pg_observations, self.pg_rewards = [], [], []
      return discounted_pg_rewards_norm


   def learn_conventional(self, episode_size, step, method = 'double'):
      sequential = False
      nForget = 50
      training_batch_size = 50  #100
      training_iteration = 40 # 200
      replace_target_iter = 20 

      if (step == episode_size - 1):
         #Drop first nForget episodes
         index_train = np.arange(nForget, episode_size)
      else:
         index_train = np.arange(0, episode_size)

      losses = []
      for i in range(training_iteration):
         np.random.shuffle(index_train)
         minibatch = self.memory[index_train[:training_batch_size]]

         q_eval = self.eval_net(torch.Tensor(minibatch[:, :self.n_features]))
         q_next = self.target_net(torch.Tensor(minibatch[:, -self.n_features:]))
         if (method == 'double'):
            q_next_action = self.eval_net(torch.Tensor(minibatch[:, :self.n_features]))
            next_action = q_next_action.max(1)[1].detach()
            # next_action = np.argmax(q_next_action, axis = 1)

         # change q_target w.r.t q_eval's action
         q_target = q_eval.detach()

         eval_act_index = minibatch[:, self.n_features].astype(int)
         reward = minibatch[:, self.n_features + 1]

         if (method == 'normal'):
            next_q_value = self.gamma * q_next.max(1)[0].detach()
            # next_q_value = self.gamma * np.max(q_next, axis=1)
            for index in range(len(eval_act_index)):
               q_target[index, eval_act_index[index]] = reward[index] + next_q_value[index]

         elif (method == 'double'):
            for index in range(len(eval_act_index)):
               q_target[index, eval_act_index[index]] = reward[index] + \
                  self.gamma * q_next[index, next_action[index]]
         self.optimizer.zero_grad()
         outputs = self.eval_net(torch.Tensor(minibatch[:, :self.n_features]))
         loss = self.criterion(outputs, q_target)
         losses.append(loss.detach().item())
         loss.backward()
         self.optimizer.step()

         if ((i+1) % replace_target_iter == 0):
            self.target_net.load_state_dict(self.eval_net.state_dict())
      # self.cost_his.append(loss.detach().item())
      # TODO: refresh last state

      ### Temporary - store memory for offline training
      minibatch = self.memory
      q_eval = self.eval_net(torch.Tensor(minibatch[:, :self.n_features]))
      q_next = self.target_net(torch.Tensor(minibatch[:, -self.n_features:]))
      if (method == 'double'):
         q_next_action = self.eval_net(torch.Tensor(minibatch[:, :self.n_features]))
         next_action = q_next_action.max(1)[1].detach()
         # next_action = np.argmax(q_next_action, axis = 1)

      # change q_target w.r.t q_eval's action
      q_target = q_eval.detach()

      eval_act_index = minibatch[:, self.n_features].astype(int)
      reward = minibatch[:, self.n_features + 1]

      if (method == 'normal'):
         next_q_value = self.gamma * q_next.max(1)[0].detach()
         # next_q_value = self.gamma * np.max(q_next, axis=1)
         for index in range(len(eval_act_index)):
            q_target[index, eval_act_index[index]] = reward[index] + next_q_value[index]

      elif (method == 'double'):
         for index in range(len(eval_act_index)):
            q_target[index, eval_act_index[index]] = reward[index] + \
               self.gamma * q_next[index, next_action[index]]
      outputs = self.eval_net(torch.Tensor(minibatch[:, :self.n_features]))
      loss = self.criterion(outputs, q_target)
      self.cost_his.append(loss.detach().item())

   def learn_PG_snn(self, step):
      discounted_pg_rewards_norm = self._pg_discount_norm_rewards()
      
      spike_inp = torch.stack(self.all_obs_spikes, dim=0)[:-1,:,:]

      self.pg_observations = np.vstack(self.pg_observations)
      
      for ep in range(pgepochs):
         self.optimizer.zero_grad()
         if REGRESSOR == 'SurrGrad':
            logits = self.run_surr_grad_snn(self.policy_net, spike_inp)
         elif REGRESSOR == 'SNN':
            logits = self.run_ncomm_snn(self.policy_net, spike_inp)
         cross_ent = F.nll_loss(F.log_softmax(logits), 
                                target=torch.LongTensor(self.pg_actions), 
                                reduction="none")
         pg_loss = torch.sum(cross_ent * torch.FloatTensor(discounted_pg_rewards_norm.copy()))
         # Entropy loss
         if ent_coef:
            policy = F.softmax(logits, dim=-1)
            log_policy = F.log_softmax(logits, dim=-1)
            entropy_loss = torch.sum(policy * log_policy)
            total_loss = pg_loss + ent_coef * entropy_loss
         else:
            total_loss = pg_loss         
            
         total_loss.backward()
         self.optimizer.step()

      self.pg_actions, self.pg_observations, self.pg_rewards = [], [], []
      self.all_obs_spikes = self.all_obs_spikes[-1:]
      return discounted_pg_rewards_norm

   def learn_snn(self, step, 
                 training_batch_size = 50, training_iteration = 100, replace_target_iter = 25,
                 debug=False, seed=1337, opt_cb=None):
      method = 'double'
      sequential = False
      nForget = 50

      spike_inp = torch.stack(self.all_obs_spikes, dim=0)
      episode_size = spike_inp.shape[0]-1

      if (step == episode_size - 1):
         #Drop first nForget episodes
         index_train = np.arange(nForget, episode_size)
      else:
         index_train = np.arange(0, episode_size)

      losses = []
      # print('shape', episode_size, self.memory.shape, training_batch_size, spike_inp.shape)
      # import pdb; pdb.set_trace()
      for i in range(training_iteration+1):
         if i == training_iteration:
            minibatch = self.memory
         else:
            np.random.shuffle(index_train)
            minibatch = self.memory[index_train[:training_batch_size]]
         # print('i', i, minibatch.shape)

         with torch.no_grad():
            if REGRESSOR == 'SurrGrad':
               q_eval = self.run_surr_grad_snn(self.eval_net, spike_inp[minibatch[:, 0],:,:])
               q_next = self.run_surr_grad_snn(self.target_net, spike_inp[minibatch[:, -1],:,:]) #TODO: 0
            elif REGRESSOR == 'SNN':
               # raise Exception('ABABAB')
               q_eval = self.run_ncomm_snn(self.eval_net, spike_inp[minibatch[:, 0],:,:])
               q_next = self.run_ncomm_snn(self.target_net, spike_inp[minibatch[:, -1],:,:]) #TODO: 0
            else: 
               raise Exception('Invalid regressor')

            if (method == 'double'):
               if REGRESSOR == 'SurrGrad':
                  q_next_action = self.run_surr_grad_snn(self.eval_net, spike_inp[minibatch[:, -1],:,:])
               elif REGRESSOR == 'SNN':
                  q_next_action = self.run_ncomm_snn(self.eval_net, spike_inp[minibatch[:, -1],:,:])
               next_action = q_next_action.max(1)[1].detach()

         # change q_target w.r.t q_eval's action
         q_target = q_eval.detach()

         eval_act_index = minibatch[:, 1].astype(int)
         reward = minibatch[:, 2]

         if (method == 'normal'):
            next_q_value = self.gamma * q_next.max(1)[0].detach()
            for index in range(len(eval_act_index)):
               q_target[index, eval_act_index[index]] = reward[index] + next_q_value[index]

         elif (method == 'double'):
            for index in range(len(eval_act_index)):
               q_target[index, eval_act_index[index]] = reward[index] + \
                  self.gamma * q_next[index, next_action[index]]
         self.optimizer.zero_grad()

         if i == training_iteration:
            torch.set_grad_enabled(False)
         if REGRESSOR == 'SurrGrad':
            outputs = self.run_surr_grad_snn(self.eval_net, spike_inp[minibatch[:, 0],:,:])
         elif REGRESSOR == 'SNN':
            outputs = self.run_ncomm_snn(self.eval_net, spike_inp[minibatch[:, 0],:,:])
         loss = self.criterion(outputs, q_target)
         losses.append(loss.detach().item())
         if i == training_iteration:
            torch.set_grad_enabled(True)
            last_loss = loss.detach().item()
         else:
            loss.backward()
         if debug == 2:
            import pdb; pdb.set_trace()
         self.optimizer.step()

         if ((i+1) % replace_target_iter == 0):
            if debug:
               print('replace target', i)
            if REGRESSOR.startswith('SurrGrad'):
               self.target_net[0] = self.eval_net[0].detach()
               self.target_net[1] = self.eval_net[1].detach()
            else:
               self.target_net.load_state_dict(self.eval_net.state_dict())

      if not debug:
         self.all_obs_spikes = [spike_inp[-1,:,:]]
         self.cost_his.append(last_loss)

      return last_loss

   def update_lr(self, lr):
      if self.agent_id == 0:
         print('* New LR:', lr)
      if RLTYPE == 'PG':
         net = self.policy_net
      elif RLTYPE == 'DQN':
         net = self.eval_net

      if REGRESSOR.startswith('SurrGrad'):
         self.optimizer = optim.Adam(net, lr=lr, betas=(0.9, 0.999))
      else:
         self.optimizer = optim.Adam(net.parameters(), lr=lr)
      #self.eval_net.lr = lr
      #self.target_net.lr = lr    