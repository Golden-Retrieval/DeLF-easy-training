# -*- coding: utf_8 -*-
#############[ Test file ]####################
"""
You have to specify
dataset directory should be
ex)
~/data/stanford_small/train/bycicle/
~/data/stanford_small/train/cup/
~/data/stanford_small/train/car/


config.data_path : ~/data/stanford_small
config.train_step : "att_learning" or "resnet_finetune"
config.save_name : where you want to save checkpoint files
config.restore_file : where your exist checkpoint files is to continue learning
config.nb_epoch : number of total epochs
config.ckpt_type : what variables restore_file have, "resnet_ckpt" or "attetion_ckpt"
"""

from delf_train import Config, DelfTrainerV1

if __name__=="__main__":
    config = Config()
    config.data_path = "/home/soma03/projects/data/stanford_small"
    # config.train_step = "resnet_finetune"
    config.train_step = "att_learning"
    config.save_name = 'local_ckpt/att_tune'
    config.restore_file = 'local_ckpt/att_tune_1+2019-03-05_20:38'
    config.nb_epoch = 2
#     config.ckpt_type = 'resnet_ckpt'
    config.ckpt_type = 'attention_ckpt'
    delf_obj = DelfTrainerV1(config)
    delf_obj.run()