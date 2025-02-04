import numpy as np
#import hickle as hkl 
from scipy.io import loadmat
from config import *

class DSA_Period():
   def __init__(self, n_channel, n_su, punish_interfer_PU = -2):
      self.n_channel = n_channel  # The number of channels
      self.n_su = n_su            # The number of SUs

      # Initialize the random seed
      self.random_seed = 7
      np.random.seed(self.random_seed)

      # Initialize the Markov channels
      self._build_periodic_channel()

      # Initialize the PU state
      self.render_PU_state()

      # Initialize the locations of SUs and PUs
      self._load_location()

      # Time subframe (1ms)
      self.t_subframe = 1

      # The subframe per sensing period
      self.subframe_per_period = 10

      # The sensed subframes per sensing period
      self.sensed_subframe_per_period = 2

      # Transmit power of PU and SU (mW)
      self.PU_power = 500
      #Replaced with POWS
      #self.SU_power = 500
      

      # Background noise
      Noise_spectral_dBm = -164 # (dBm/Hz)
      self.B = 5 # (MHz)
      Noise_dBm = Noise_spectral_dBm + 10 * np.log10(self.B * (10 ** 6)) # (dBm)
      self.Noise = 10 ** (Noise_dBm / 10) # (mW)

      # Load channels
      self._load_channel()

      # The punishment for interfering PUs
      self.punish_interfer_PU = punish_interfer_PU

      # The bps threshold for interfering PUs
      self.warning_threshold = 1.5

      # The bps threshold for SUs' collision
      self.SU_collision_threshold = 1.5

   def _build_periodic_channel(self):
      self.channel_state = np.zeros(self.n_channel)
      # Initialize period
      #self.period = np.array([3, 4, 5, 3, 4, 5])
      if self.n_channel == 6:
         self.period = np.array([2, 3, 4, 2, 3, 4])
      if self.n_channel == 4:
         self.period = np.array([3, 4, 3, 4])
      self.count = np.random.randint(4, size=self.n_channel)

   def render_PU_state(self):
      self.count = self.count + 1
      for k in range(self.n_channel):
         self.channel_state[k] = 0
         if (self.count[k] % self.period[k] == 0):
            self.channel_state[k] = 1

   def _load_channel(self):
      # filename = 'winner2_channel_PU%d_SU%d.mat' % (self.n_channel, self.n_su)
      try:         
         filename = 'winner2_channel_PU%d_SU%d.hkl' % (self.n_channel, self.n_su)
         mat = hkl.load(filename)
         self.H_PUR_PUT = mat['H_PUR_PUT']
      except:
         filename = 'winner2_channel_PU%d_SU%d.npz' % (self.n_channel, self.n_su)
         mat = np.load(filename)
         self.H_PUR_PUT = mat['H_PUR_PUT']         

      #plt.figure()
      #time = np.arange(10001)
      #plt.plot(time/1000, np.absolute(self.H_PUR_PUT[0, time]), 'b-')
      #plt.ylabel('Amplitude of channel gain')
      #plt.xlabel('Time (* 1s)')
      #plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
      #plt.xlim(0, 10)
      #plt.show()
      self.H_SUR_SUT = mat['H_SUR_SUT']
      self.H_SUR_PUT = mat['H_SUR_PUT']
      self.H_PUR_SUT = mat['H_PUR_SUT']
      self.H_SUT_PUT = mat['H_SUT_PUT']

   def _load_location(self):
      # Load locations of PUs and SUs
      filename = 'location_PU%d_SU%d.mat' % (self.n_channel, self.n_su)
      self.PUT_loc = loadmat(filename)['PUT_loc']
      self.PUR_loc = loadmat(filename)['PUR_loc']
      self.SUT_loc = loadmat(filename)['SUT_loc']
      self.SUR_loc = loadmat(filename)['SUR_loc']
      self.area = loadmat(filename)['rmax']
      #self._plot_location()

   def _plot_location(self):
      # Plot locations of PUs and SUs
      plt.figure(figsize=(6.6, 6.6))


      plt.plot(self.PUT_loc[0, :], self.PUT_loc[1, :], 'ro', label='PUT')
      plt.plot(self.PUR_loc[0, :], self.PUR_loc[1, :], 'rx', label='PUR')
      plt.plot(self.SUT_loc[0, :], self.SUT_loc[1, :], 'bs', label='SUT')
      plt.plot(self.SUR_loc[0, :], self.SUR_loc[1, :], 'b^', label='SUR')

      for n in range(self.n_su):
         labelstr = "SUT%d" % (n+1)
         plt.annotate(
               labelstr,
          xy=(self.SUT_loc[0, n], self.SUT_loc[1, n]), xytext=(0, 4),
           textcoords='offset points', ha='center', va='bottom',
            )
         labelstr = "SUR%d" % (n+1)
         plt.annotate(
               labelstr,
          xy=(self.SUR_loc[0, n], self.SUR_loc[1, n]), xytext=(0, 4),
           textcoords='offset points', ha='center', va='bottom',
            )

      for n in range(self.n_channel):
         labelstr = "PUT%d" % (n+1)
         plt.annotate(
               labelstr,
          xy=(self.PUT_loc[0, n], self.PUT_loc[1, n]), xytext=(0, 4),
           textcoords='offset points', ha='center', va='bottom',
            )
         labelstr = "PUR%d" % (n+1)
         plt.annotate(
               labelstr,
          xy=(self.PUR_loc[0, n], self.PUR_loc[1, n]), xytext=(0, 4),
           textcoords='offset points', ha='center', va='bottom',
            )
      plt.legend(loc='lower left')
      plt.ylabel('y')
      plt.xlabel('x')
      plt.xlim(0, self.area)
      plt.ylim(0, self.area)


   def _quantize_reward(self, dataRate):
      bps = dataRate/self.B
      if (bps < 1):
         reward = 0
      elif (bps >= 1 and bps < 2):
         reward = 1
      elif (bps >= 2 and bps < 3):
         reward = 2
      else:
         reward = 3
      return reward

   def access(self, action, period):
      self.success_SU = np.zeros(self.n_su)  # a SU doesn't collide with other SUs or degrades PU data rate
      self.fail_PU = np.zeros(self.n_channel) # a SU collides with a PU and degrades PU data rate
      self.fail_SU = np.zeros(self.n_su) # a SU collides with other SUs
      self.access_SU = np.zeros(self.n_su) # the number of SUs' access
      self.power_SU = np.zeros(self.n_su)
      self.access_channel_SU = - np.ones(self.n_su)

      self.reward = - np.ones(self.n_su)
      self.dataRate_SU = np.zeros(self.n_su)
      self.dataRate_PU = np.zeros(self.n_channel)

      # Define the time for access period
      time = np.arange(self.subframe_per_period * period + self.sensed_subframe_per_period,
                         self.subframe_per_period * (period + 1))

      # Calculate the data rate of PU
      for n in range(self.n_channel):
         if (self.channel_state[n] == 1):
            H_PUR_PUT_power = np.absolute(self.H_PUR_PUT[n, time]) ** 2

            # Calculate the interference between PUR/SUT
            Interferecne_PUR_SUT = np.zeros(self.subframe_per_period - self.sensed_subframe_per_period)
            
            interfered_SUT = np.where(((nPOWS+1)*n <= action) & (action < (nPOWS+1)*(n+1)-1))[0]
            #interfered_SUT = np.where(action == 2*n)[0]
            for m in interfered_SUT:
               H_PUR_SUT_power = np.absolute(self.H_PUR_SUT[n, m, time]) ** 2
               Interferecne_PUR_SUT += H_PUR_SUT_power * POWS[action[m]%(nPOWS+1)] # * self.SU_power

            SINR = H_PUR_PUT_power * self.PU_power / (Interferecne_PUR_SUT + self.Noise)

            self.dataRate_PU[n] = np.mean(self._calculate_dataRate(SINR))

      # Calculate the data rate of SU
      for k in range(self.n_su):
         #if ((action[k] % nPOWS) == 1): # action is not accessing any channel
         if ((action[k] % (nPOWS+1)) == nPOWS): # action is not accessing any channel
            self.dataRate_SU[k] = 0
            self.power_SU[k] = 0

         else: # action is accessing one channel

            access_channel = action[k] // (nPOWS+1)
            self.power_SU[k] = POWS[action[k] % (nPOWS+1)]

            self.access_channel_SU[k] = access_channel

            H_SUR_SUT_power = np.absolute(self.H_SUR_SUT[k, k, time]) ** 2

            # Calculate the interference between SUR/SUT
            Interferecne_SUR_SUT = np.zeros(self.subframe_per_period - self.sensed_subframe_per_period)
            #interfered_SUT = np.where(action == action[k])[0]
            interfered_SUT = np.where(((nPOWS+1)*(action[k]//nPOWS) <= action) & (action < (nPOWS+1)*(action[k]//nPOWS+1)-1))[0]
            interfered_SUT = interfered_SUT[interfered_SUT != k] # except itself
            for m in interfered_SUT:
               H_SUR_SUT_power = np.absolute(self.H_SUR_SUT[k, m, time]) ** 2
               Interferecne_SUR_SUT += H_SUR_SUT_power * POWS[action[m]%(nPOWS+1)] # * self.SU_power

            # Calculate the interference between SUR/PUT
            if self.channel_state[access_channel] == 1:
               H_SUR_PUT_power = np.absolute(self.H_SUR_PUT[k, access_channel, time]) ** 2
               Interferecne_SUR_PUT = H_SUR_PUT_power * self.PU_power
            else:
               Interferecne_SUR_PUT = 0

            # Calculate the total interference
            Interferecne = Interferecne_SUR_SUT + Interferecne_SUR_PUT

            #SINR = H_SUR_SUT_power * self.SU_power / (Interferecne + self.Noise)
            SINR = H_SUR_SUT_power * POWS[action[k]%(nPOWS+1)] / (Interferecne + self.Noise)

            self.dataRate_SU[k] = np.mean(self._calculate_dataRate(SINR))

      # Calculate the reward of SU
      for k in range(self.n_su):
         if ((action[k] % (nPOWS+1)) == nPOWS):  # action is not accessing any channel
            self.reward[k] = -1
            self.access_SU[k] = 0
         else:  # action is accessing one channel

            access_channel = action[k] // (nPOWS+1) #int(action[k] / 2)

            self.access_SU[k] = 1

            # Check if SU collides with PU and causes strong interference
            if self.channel_state[access_channel] == 1:
               if (self.dataRate_PU[access_channel] / self.B) < self.warning_threshold:
                  # collision with PU
                  self.fail_PU[access_channel] = 1
                  self.fail_SU[k] = 1
                  self.reward[k] = self.punish_interfer_PU
               else:
                  self.fail_PU[access_channel] = 0
                  self.reward[k] = self._quantize_reward(self.dataRate_SU[k])
                  if REW_TYPE == 'PowPenalty':
                     self.reward[k] -= powPenalty[action[k]%(nPOWS+1)]
            else:
               self.fail_PU[access_channel] = 0
               self.reward[k] = self._quantize_reward(self.dataRate_SU[k])
               if REW_TYPE == 'PowPenalty':
                  self.reward[k] -= powPenalty[action[k]%(nPOWS+1)]               

            # Check if SU collides with other SUs
            #interfered_SUT = np.where(action == action[k])[0]
            interfered_SUT = np.where(((nPOWS+1)*(action[k]//nPOWS) <= action) & (action < (nPOWS+1)*(action[k]//nPOWS+1)-1))[0]
            
            interfered_SUT = interfered_SUT[interfered_SUT != k]
            if (len(interfered_SUT) > 0):
               if (self.dataRate_SU[k] / self.B) < self.SU_collision_threshold:
                  # collision with other SUs
                  self.fail_SU[k] = 1
               elif self.fail_PU[access_channel] == 0:
                  # successful transmission
                  self.success_SU[k] = 1
            elif self.fail_PU[access_channel] == 0:
               # successful transmission
               self.success_SU[k] = 1

      return self.reward

   def _calculate_dataRate(self, SINR):
      dataRate = np.zeros(SINR.shape) # Kbps
      SINR_dB = 10 * np.log10(SINR)

      SINR_dB_level = [-6.9360, -5.1470, -3.1800, -1.2530, 0.7610, 2.6990, 4.6940, 6.5250, 8.5730, 10.3660,
                         12.2890, 14.1730, 15.8880, 17.8140, 19.8290]
      bps_level = [0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141, 2.4063, 2.7305,
                     3.3223, 3.9023, 4.5234, 5.1152, 5.5547]

      for k in range(SINR_dB.size):
         for i in range(15):
            if (i != 14):
               if SINR_dB[k] >= SINR_dB_level[i] and SINR_dB[k] < SINR_dB_level[i+1]:
                  dataRate[k] = bps_level[i] * self.B
                  break
            else:
               if SINR_dB[k] >= SINR_dB_level[i]:
                  dataRate[k] = bps_level[i] * self.B

      return dataRate

   def render_sensor(self, action):
      active_sensor = np.zeros((self.n_su, self.n_channel)).astype(np.int32)
      #initial_sensed_channel = np.floor(action / 2).astype(np.int32)
      initial_sensed_channel = action // (nPOWS+1)
      for k in range(self.n_su):
         active_sensor[k, initial_sensed_channel[k]] = 1
      return active_sensor

   def sense(self, active_sensor, period):
      time = np.arange(self.subframe_per_period * period,
                         self.subframe_per_period * period + self.sensed_subframe_per_period)

      # Calculate the sensed PU signal at SUT
      H_SUT_PUT = self.H_SUT_PUT[:, :, time]
      channel_state = np.array([self.channel_state for k in range(self.n_su)])
      sensed_PU_signal = (self.PU_power ** 0.5) * H_SUT_PUT
      for t in range(self.sensed_subframe_per_period):
         sensed_PU_signal[:, :, t] = sensed_PU_signal[:, :, t] * channel_state

      # The number of samples per millisecond
      symbol_per_subframe = 14

      threshold = np.zeros((self.n_su, 1 + self.n_channel))

      # const = 5 * (10 ** 8) # normalize the sensed signal
      const = (10 ** 6)  # normalize the sensed signal

      for n in range(self.n_su):
         for t in range(self.sensed_subframe_per_period):
            c = np.where(active_sensor[n, :] == 1)[0]
            Noise_real = np.random.normal(0, (self.Noise / 2) ** 0.5, symbol_per_subframe)
            Noise_imag = np.random.normal(0, (self.Noise / 2) ** 0.5, symbol_per_subframe)
            Noise = Noise_real + 1j * Noise_imag
            sensed_PU_signal_repeat = np.repeat(sensed_PU_signal[n, c, t], symbol_per_subframe)
            sensed_signal = sensed_PU_signal_repeat + Noise
            sensed_signal_power = np.absolute(sensed_signal) ** 2
            sensed_signal_power_sum = np.sum(sensed_signal_power)
            threshold[n, 0] = threshold[n, 0] + sensed_signal_power_sum * const

      threshold[:, 0] = threshold[:, 0] / (self.sensed_subframe_per_period * symbol_per_subframe)
      #threshold[:, 0] = (threshold[:, 0] <= 1) * threshold[:, 0] + (threshold[:, 0] > 1) * np.ones(self.n_su)

      # sensed channel indicator
      threshold[:, 1:(self.n_channel+1)] = active_sensor

      return threshold    