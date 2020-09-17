import train.trainer as trainer
import predict.predictor as vnet3d_predictor

def test():
    case_root_path = 'D:\\tmp\\MIS\\hippo\\train'
    log_root_path = 'D:\\tmp\\MIS\\hippo\\log'
    model_root_path = 'D:\\tmp\\MIS\\hippo\\model'
    loc_root_path = 'D:\\tmp\\MIS\\hippo\\location'
    modality = 'MR'
    vnet3d_trainer = trainer.VNet3DTrainer(case_root_path, model_root_path, log_root_path)
    vnet3d_trainer.train_location(modality)
    vnet3d_predictor.predict_location(modality, case_root_path, model_root_path, loc_root_path)
    # vnet3d_trainer.train_segmentation(modality, loc_root_path)
    pass

if __name__ == '__main__':
    test()
    print('trainer test end.')
    pass