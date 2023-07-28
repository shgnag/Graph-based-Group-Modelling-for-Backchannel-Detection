import torch.nn as nn
import torch.optim as optim


model_inputs = {
    # input files
    'train_type': 'bck', #['bck', 'agr'] for backchannel and agreement
    'training_mode': 'Static_abstract', #['Static_abstract', 'Dynamic_abstract','Static_deep','Dynamic_deep']

    'features_head': 'intermediate_results/backchannel_training_OpenfaceHGD_ALLPersons.csv',
    'features_pose': 'intermediate_results/backchannel_training_ALLPersons_POSE.csv',
    'featuresOF_main': 'backchannel/HGD_annotated_features_Main',
    'featuresOF_context': 'backchannel/HGD_annotated_features_context',
    'feature_openpose_main': 'backchannel/openpose_processed',
    'feature_openpose_con': '/backchannel/openpose_light',
    
    'train_id_file': 'backchannel_train_main/main/labels/bc_detection_train.csv',
    'valid_id_file': 'backchannel_train_main/main/labels/bc_detection_val.csv',
    'test_id_file': 'backchannel_main_test/main/labels/bc_detection_test.csv',
    
    'train_id_agr_file': 'backchannel_train_main/main/labels/bc_agreement_train.csv',
    'valid_id_agr_file': 'backchannel_train_main/main/labels/bc_agreement_val.csv',
    'test_id_agr_file': 'backchannel_main_test/main/labels/bc_agreement_test.csv',
    
    'features_test_head': '/intermediate_results/backchannel_HEAD_ALLPersonsTEST.csv',
    'features_test_pose': 'intermediate_results/backchannel_POSE_ALLPersonsTEST.csv',

    'DeepAV_Feats': '/backchannel/DeepAV_Features',
    'DeepAV_Feats_test': 'backchannel/DeepAV_Features_TEST',
    'models_out': 'GroupGraph/models'
}
model_optimization_params = {
    # Optimizer config
    'optimizer': optim.Adam, 
    'criterion': nn.CrossEntropyLoss(),#nn.MSELoss,#nn.CrossEntropyLoss(), 
    'learning_rate': 3e-2,
    'epochs': 50,
    'step_size': 7,
    'gamma': 0.1,

    # Batch Config
    'batch_size': 8,
    'threads': 16
}
