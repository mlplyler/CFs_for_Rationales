Pretrain two types of transformers. This was done on multiple GPUs in a SLURM environment. You may find ../utils/pre_train_multi.py helpful.
  To pretrain the random masking model
    python pretrain_randmask.py ../configs/loc_rand_config.json
  To pretrain the contiguous mask model
    python pretrain_contmask.py ../configs/loc_cont_config.json
  We cleaned up the subsequent models directors so that cfmodel, cfmodel0, and cfmodel1 were all the same. We also took the directory with the last checkpoint.

Train a rationale model (MMI)
  python make_chkpt.py ../configs/loc_rat_config.json
  python train_rationale.py ../models/MMI/config.json

Finetune the rationale model classifier
  python train_finetune_rationale.py ../models/MMI/config.json

Train the counterfactual predictor models
  python train_CFPredictor.py ../configs/loc_cfpredictor_positive_config.json
  python train_CFPredictor.py ../configs/loc_cfpredictor_negative_config.json

Dump the counterfactuals
  cd ../utils/
  python dump_CFs.py ../models/both_CFpredictors/ argmax_train.dump -randsamp 0 -td train -flipit 1 -iterdecode 1

Make a training file
  python make_CF_file.py ../models/both_CFpredictors/argmax_train.dump 0 1

We shuffled the resultant file
  shuf ../models/both_CFpredictors/argmax_train.dump.ratform > ../models/both_CFpredictors/argmax_train.dump.ratform.shuf

Train the CDA rationale model
  python make_chkpt.py ../configs/loc_cda_config.json
  python train_rationale.py ../models/loc_cda_config.json
  python train_finetune_rationale.py ../models/loc_cda_config.json

Test the models with 
  python test_rationale.py 1 ../models/loc_cda_config.json


The shufs will potentially throw off the downstream numbers. 
You can do FDA with -flipit 0.
You can get the antonym baseline in ../utils/


