import torch
import os, sys
import platform
import base64

if len(sys.argv) > 1:
   print('Loading config from argv.')
   try:
      loaded_dict = eval(base64.b64decode(sys.argv[1]))
      for k, v in loaded_dict.items():
         print('{}: {}'.format(k, v))
         os.environ[k] = str(v)
   except Exception:
      print("Failed to load from argv!")

#os.environ["LSM"] = '0'
#os.environ["REGRESSOR"] = 'SNN'
#os.environ["ent_coef"] = '0.05'

n_channel = 4
n_su = 6

device = os.getenv('device', 'cpu')
print('*, device')

if device == 'gpu':
   if torch.cuda.is_available():
      device = torch.device("cuda")     
   else:
      device = torch.device("cpu")
else:
   device = torch.device("cpu")

print('Running on', device.type)

def get_slurm_env(type_, name, default):
   res = os.getenv(name, str(default)).replace("'","")
   if type_ == 'int':
      res = int(res)
   elif type_ == 'float':
      res = float(res)
   elif type_ == 'bool':
      res = bool(int(res))
   return res


fname_list = []
conf = {}

#Scenario: DSS (Ch. switching), SSSD (discrete), SSSC (continuous)
SCENARIO = get_slurm_env('str', 'SCENARIO', 'SSSD') 
fname_list.append('00{}'.format(SCENARIO))

REW_TYPE = get_slurm_env('str', 'REW_TYPE', 'PowPenalty') # Orig, PowPenalty
fname_list.append('21{}'.format(REW_TYPE))

RLTYPE = get_slurm_env('str', 'RLTYPE', 'PG') #DQN, PG
fname_list.append('01{}'.format(RLTYPE))
conf['RLTYPE'] = RLTYPE

REGRESSOR = get_slurm_env('str', 'REGRESSOR', 'MLP') #LinReg, MLP, SNN, SNN_scaled, LSM, SurrGrad
fname_list.append('02{}'.format(REGRESSOR))
conf['REGRESSOR'] = REGRESSOR

learning_rate = get_slurm_env('float', 'LR', '0.05')
fname_list.append('05LR{:.4f}'.format(learning_rate))
conf['learning_rate'] = learning_rate

POWS = get_slurm_env('str', 'POWS', '100-200-300')
POWS = list(map(int, POWS.split('-')))
nPOWS = len(POWS)

if SCENARIO in ('SSSD', 'SSSC'):
   fname_list.append('20POW{}'.format(len(POWS)))
elif SCENARIO == 'DSS':
   POWS = [500]
   nPOWS = len(POWS)
conf['POWER_LEVELS'] = POWS
conf['POWER_LEVELS_num'] = nPOWS
powPenalty = torch.hstack([torch.linspace(0, 1, nPOWS+1)[1:], torch.Tensor([0])]).tolist()


if RLTYPE == 'DQN':
   tbs = get_slurm_env('int', 'tbs', 35)
   fname_list.append('06tbs{}'.format(tbs))
   conf['tbs'] = tbs
   ti = get_slurm_env('int', 'ti', 90)
   fname_list.append('07ti{}'.format(ti))
   conf['ti'] = ti
   rti = get_slurm_env('int', 'rti', 30)
   fname_list.append('08rti{}'.format(rti))
   conf['rti'] = rti
   
elif RLTYPE == 'PG':
   pgepochs= get_slurm_env('int', 'pgepochs', 5)
   fname_list.append('09pgepochs{}'.format(pgepochs))
   conf['pgepochs'] = pgepochs
   ent_coef= get_slurm_env('float', 'ent_coef', 0)
   fname_list.append('19ent_coef{}'.format(ent_coef))
   conf['ent_coef'] = ent_coef
else:
   raise Exception('invalid RLTYPE')

if REGRESSOR in ['MLP', 'SurrGrad', 'SNN']:
   hidden = get_slurm_env('int', 'hidden', 10)
   fname_list.append('18hidden{}'.format(hidden))
   conf['hidden'] = hidden

USE_LSM = get_slurm_env('bool', 'LSM', 1)
if REGRESSOR in ['LinReg', 'MLP']:
   USE_LSM = False
   conf['USE_LSM'] = False
   
elif REGRESSOR in ['SurrGrad', 'SNN']:
   ENCODER = get_slurm_env('str', 'ENCODER', 'ISI') # Poisson, ISI, Phase+ISI
   fname_list.append('03{}'.format(ENCODER))
   conf['ENCODER'] = ENCODER
   #-
   CONV_TYPE = get_slurm_env('int', 'CONV', 3)
   fname_list.append('04CONV{}'.format(CONV_TYPE))
   conf['CONV_TYPE'] = CONV_TYPE
   if USE_LSM:
      conf['USE_LSM'] = True
      fname_list.append('10LSM')
      minicol = get_slurm_env('str', 'minicol', '2-2-2')
      minicol = list(map(int, minicol.split('-')))
      fname_list.append('13Mini{}'.format('_'.join([str(_) for _ in minicol])))
      conf['minicol'] = minicol
      #-
      macrocol = get_slurm_env('str', 'macrocol', '2-2-2')
      macrocol = list(map(int, macrocol.split('-')))
      fname_list.append('14Macro{}'.format('_'.join([str(_) for _ in macrocol])))
      conf['macrocol'] = macrocol
      #-
      SpecRAD = get_slurm_env('bool', 'SpecRAD', 0)
      fname_list.append('11SR{:d}'.format(SpecRAD))
      conf['SpecRAD'] = SpecRAD
      #-
      PMAX = get_slurm_env('float', 'PMAX', '0.1')
      fname_list.append('12PMAX{:.3f}'.format(PMAX))
      conf['PMAX'] = PMAX
      #-
      ALPHA = get_slurm_env('float', 'ALPHA', '0.01')
      fname_list.append('15Alp{:.4f}'.format(ALPHA))
      conf['ALPHA'] = ALPHA
      #-
      readout_inp = get_slurm_env('int', 'readoutinp', '32')
      fname_list.append('16lsminp{}'.format(readout_inp))
      conf['lsminp'] = readout_inp
      #-
      readout_out = get_slurm_env('int', 'readoutout', '16')
      fname_list.append('17lsmout{}'.format(readout_out))
      conf['lsmout'] = readout_out
   else:
      fname_list.append('10NoLSM')


fname_list = sorted(fname_list)
FNAME = '_-_'.join(fname_list)
print(FNAME)

RESULT_PATH = '/content/drive/MyDrive/TNNLS/LAST/'
if not os.path.exists(RESULT_PATH):
   RESULT_PATH = './results/'
   
if os.path.exists(os.path.join(RESULT_PATH, '{}_PU{}_SU{}.hkl'.format(FNAME, n_channel, n_su))):
   if os.getenv('dryrun', False):
      print("Dryrun!")
   elif 'Darwin' not in platform.system():
      print('Already had executed this configuration. Exiting...')
      sys.exit(1)
   else:
      print('Debugging on macos')
   
with open(os.path.join(RESULT_PATH, '{}_PU{}_SU{}.hkl'.format(FNAME, n_channel, n_su)), 'wb') as fo:
   fo.write('X'.encode("ascii"))
