import os
import time
import random
import argparse
import json



if __name__=='__main__':
  ######## load and parse args
  parser = argparse.ArgumentParser()
  parser.add_argument('thedir')
  targs = parser.parse_args()  
  source_dir = '../models/mfix/{}/torun/'.format(targs.thedir)
  save_dir = '../models/mfix/{}/dunrun/'.format(targs.thedir)
  if not os.path.exists(source_dir):
    print('MISSING', source_dir)
    exit()
  todo=1
  while todo:
    todo = 0
    thefiles = list(os.listdir(source_dir))
    random.shuffle(thefiles)
    print('THEFILES', thefiles)
    for f in thefiles:
      if 'json' in f and f not in os.listdir(save_dir):# and '3' in f:
        with open(source_dir+f,'r') as f1:
          cstr = f1.read()
        args = json.loads(cstr)
        log_path = args['log_path']
        os.system('mv {} {}'.format(source_dir+f,save_dir+f))
        ## this makes the config.json
        os.system('cd /home/mlplyler/ratgit/mfix/ && python chkpt_fix.py {}'.format(save_dir+f))
        ## load the config.json
        os.system(
        'cd ../train/ && python /home/mlplyler/ratgit/mfix/train_rationale.py {}'.format(
          log_path+'config.json'))        
        todo = 1      
        os.system('cd ../train/ && python test_rationale.py {} 0'.format(log_path+'config.json'))
        
