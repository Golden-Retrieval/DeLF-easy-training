from delf_train import Config, DelfTrainerV1

if __name__=="__main__":
    config = Config()
    config.data_path = "/home/soma03/projects/data/paris"
    config.train_step = "resnet_finetune"
#     config.train_step = "att_learning"
    config.save_name = 'local_ckpt/resnet_finetune'
    config.restore_file = 'resnet_v1_50.ckpt'
    config.nb_epoch = 100
    config.batch_size = 64 
    config.ckpt_type = 'resnet_ckpt'
#     config.ckpt_type = 'attention_ckpt'
    delf_obj = DelfTrainerV1(config)
    delf_obj.run()
