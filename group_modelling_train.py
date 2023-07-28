import sys
import torch

import config as exp_conf
import models_group_graph as model

from torch.optim import lr_scheduler
from torch_geometric.data import DataLoader

from datasets import *
from optimization import optimize_group_graph, _get_prediction
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # experiment Reproducibility
    torch.manual_seed(11)
    torch.cuda.manual_seed(22)
    torch.backends.cudnn.deterministic = True
    print(sys.argv)


    # CUDA_LAUNCH_BLOCKING=1
    time = 1# static specified num of frames to be included in case of dynamic
    num_nodes = 4#  implies num of nodes in one graph
    hidden_size = 64

    io_config = exp_conf.model_inputs
    opt_config = exp_conf.model_optimization_params

    # io config
    model_name = 'Model_'+io_config['train_type']+'_'+io_config['training_mode']+'_time'+str(time)+'_dim_'+str(hidden_size)

    op_file = 'predictions_' + io_config['train_type']+'_'+io_config['training_mode']+'_'+str(num_nodes)+'nodes.csv' 
    target_models = io_config['models_out']

    # cuda config
    model = model.GroupGraph(32, hidden_size) 

    # GPU stuff
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using ', device)
    model.to(device)

    criterion = opt_config['criterion']
    optimizer = opt_config['optimizer'](model.parameters(),
                                        lr=opt_config['learning_rate'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt_config['step_size'],
                                    gamma=opt_config['gamma'])

    if io_config['training_mode'] == 'Static_abstract':
        # Static graph + abstract features
        dataset_train   = MPIIDatasetStaticMain(io_config['train_id_agr_file'],io_config['features_head'],io_config['features_pose'],num_nodes)
        dataset_val = MPIIDatasetStaticMain(io_config['valid_id_agr_file'],io_config['features_head'],io_config['features_pose'],num_nodes)
        dataset_test = MPIIDatasetStaticMain(io_config['test_id_agr_file'],io_config['features_test_head'],io_config['features_test_pose'],num_nodes)

    elif io_config['training_mode'] == 'Dynamic_abstract':
        # Dynamic graph + abstract features
        dataset_train   = MPIIDatasetTemporalMain(io_config['train_id_agr_file'],io_config['featuresOF_main'],
                            io_config['featuresOF_context'],io_config['feature_openpose_main'],io_config['feature_openpose_con'], time, num_nodes)
        dataset_val = MPIIDatasetTemporalMain(io_config['valid_id_agr_file'],io_config['featuresOF_main'],
                            io_config['featuresOF_context'],io_config['feature_openpose_main'],io_config['feature_openpose_con'], time, num_nodes)

    elif io_config['training_mode'] == 'Static_deep':
        # static graph + AV deep extracted features
        dataset_train   = MPIIDatasetAVFeats_Static(io_config['train_id_agr_file'],io_config['DeepAV_Feats'],num_nodes)
        dataset_val = MPIIDatasetAVFeats_Static(io_config['valid_id_agr_file'],io_config['DeepAV_Feats'],num_nodes)
        dataset_test = MPIIDatasetAVFeats_Static(io_config['test_id_agr_file'],io_config['DeepAV_Feats_test'],num_nodes)
    
    elif io_config['training_mode'] == 'Dynamic_deep':
        # Dynamic graph + AV deep extracted features
        dataset_train   = MPIIDatasetAVFeats_Dynamic(io_config['train_id_agr_file'],io_config['DeepAV_Feats'],time,num_nodes)
        dataset_val = MPIIDatasetAVFeats_Dynamic(io_config['valid_id_agr_file'],io_config['DeepAV_Feats'],time,num_nodes)
        dataset_test = MPIIDatasetAVFeats_Dynamic(io_config['test_id_agr_file'],io_config['DeepAV_Feats_test'],time,num_nodes)
    

    dl_train = DataLoader(dataset_train, batch_size=opt_config['batch_size'],
                          shuffle=True, num_workers=opt_config['threads'])
    dl_val = DataLoader(dataset_val, batch_size=opt_config['batch_size'],
                        shuffle=False, num_workers=opt_config['threads'])
    dl_test = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    optimize_group_graph(model, (num_nodes, time), dl_train, dl_val, device,
                             criterion, optimizer, scheduler,opt_config['batch_size'],
                             num_epochs=opt_config['epochs'],
                             models_out=target_models, train_type = io_config['train_type'])

    # model_path = 'models/Model_bck_Static_abstract_time1_dim_64/0.0844067.pth'
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint)
    # _get_prediction(model, (1, num_nodes, time_length), dl_test, criterion, device, train_type = 'bck', output_file = op_file)
