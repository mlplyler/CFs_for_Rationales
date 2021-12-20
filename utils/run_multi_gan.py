import os
import time
import random
import argparse


if __name__=='__main__':
  ######## load and parse args
  parser = argparse.ArgumentParser()
  parser.add_argument('thedir')
  targs = parser.parse_args()  
  source_dir = '../models/models/fcbert/{}/torun/'.format(targs.thedir)
  save_dir = '../models/models/fcbert/{}/dunrun/'.format(targs.thedir)
  if not os.path.exists(source_dir):
    exit()
  todo=1
  while todo:
    todo = 0
    thefiles = list(os.listdir(source_dir))
    random.shuffle(thefiles)
    print('THEFILES', thefiles)
    for f in thefiles:
      if 'json' in f and f not in os.listdir(save_dir):# and '3' in f:
        print('HOSTNAME', os.environ['HOSTNAME'])
        os.system('mv {} {}'.format(source_dir+f,save_dir+f))
        os.system('cd ../train/ && python /home/mlplyler/ratgit/mfix/train_GAN.py {}'.format(save_dir+f))

        todo = 1
        time.sleep(5)
