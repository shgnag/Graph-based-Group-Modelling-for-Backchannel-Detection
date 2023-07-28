import os
import torch
import random

import numpy as np
import core.io as io
import core.clip_utils as cu
import multiprocessing as mp
from sklearn import preprocessing
from torch.utils import data
from torch_geometric.data import Data
import pandas as pd

# Static graph + abstract features
class MPIIDatasetStaticMain(data.Dataset):
# MPIIDatasetStaticMain as edge matric is created by creating a bidirecitonal edge between main and context persons
    def __init__(self, file_info,feature_path_vid,feature_path_pose, nodes):
        super(MPIIDatasetStaticMain, self).__init__()
        info = pd.read_csv(file_info) 
        self.file_name = info['id']
        self.bc_label = info['label']
        self.num_nodes = nodes
        df = pd.read_csv(feature_path_vid) 
        # self.feature_path_video = df.fillna(0)#debug

        dfp = pd.read_csv(feature_path_pose)
        # self.feature_path_pose = dfp.fillna(0) #debug

    def __getitem__(self, index):
        
        filename = self.file_name[index] #0_train_rec18_pos3
        # print('filename',filename)
        main_pos = int(filename.split('_pos')[1])-1 # index of main not the main pos

        # Checking and extracting the relevant features
        # feature_path_video file contains the man and meanDiff features of all the persons
        feature_cols = [x for x in self.feature_path_video if 'main' in x] #main person
        feature_main = self.feature_path_video[feature_cols]
        feature_main = feature_main.values[int(filename.split('_')[0]),:] 

        # pose
        # feature_cols = [x for x in self.feature_path_pose if 'main' in x]
        # ft_p_main = self.feature_path_pose[feature_cols]
        # ft_p_main = ft_p_main.values[int(filename.split('_')[0]),:]
        # feature_main = np.concatenate((feature_main,ft_p_main), axis = 0)

        feature_main = np.expand_dims(feature_main, axis=0)
        # print(feature_main)

        feature_cols = [x for x in self.feature_path_video if 'con1' in x] # context person 1
        feature_con1 = self.feature_path_video[feature_cols]
        feature_con1 = feature_con1.values[int(filename.split('_')[0]),:]
        # pose
        # feature_cols = [x for x in self.feature_path_pose if 'con1' in x]
        # ft_p_con1 = self.feature_path_pose[feature_cols]
        # ft_p_con1 = ft_p_con1.values[int(filename.split('_')[0]),:]
        # feature_con1 = np.concatenate((feature_con1,ft_p_con1), axis = 0)
        feature_con1 = np.expand_dims(feature_con1, axis=0)
        
        feature_cols = [x for x in self.feature_path_video if 'con2' in x] #context person 2
        feature_con2 = self.feature_path_video[feature_cols]
        feature_con2 = feature_con2.values[int(filename.split('_')[0]),:]
        #pose
        # feature_cols = [x for x in self.feature_path_pose if 'con2' in x]
        # ft_p_con2 = self.feature_path_pose[feature_cols]
        # ft_p_con2 = ft_p_con2.values[int(filename.split('_')[0]),:]
        # feature_con2 = np.concatenate((feature_con2,ft_p_con2), axis = 0)
        feature_con2 = np.expand_dims(feature_con2, axis=0)
        
        feature_cols = [x for x in self.feature_path_video if 'con3' in x] #context person 3
        feature_con3 = self.feature_path_video[feature_cols]
        feature_con3 = feature_con3.values[int(filename.split('_')[0]),:]
        # pose - if need to be included
        # feature_cols = [x for x in self.feature_path_pose if 'con3' in x]
        # ft_p_con3 = self.feature_path_pose[feature_cols]
        # ft_p_con3 = ft_p_con3.values[int(filename.split('_')[0]),:]
        # feature_con3 = np.concatenate((feature_con3,ft_p_con3), axis = 0)
        feature_con3 = np.expand_dims(feature_con3, axis=0)

        # Feature dim is 3 for head features only (no pose) and 46 for head + pose
        feature_set_vid = np.concatenate([feature_main, feature_con1,feature_con2,feature_con3], axis = 0) #4, 46

        # adj matrix
        # Main person connexted to all other nodes by bidirectional edges
        src = []
        dst = []
        all_p = np.arange(self.num_nodes)
        con_id = np.setdiff1d(all_p, main_pos)

        src.extend([0,0,0,1,2,3])
        dst.extend([1,2,3,0,0,0])
        # diff ways to create an adj matrix are evaluated


        batch_edges = torch.tensor([src, dst], dtype=torch.long)
        bc_label = self.bc_label[index]

        target_set = np.asarray([0, 0, 0,0]).astype(float)
        target_set[:] = bc_label #same labels for all nodes


        feature_set =  feature_set_vid
        scaler = preprocessing.StandardScaler().fit(feature_set)
        feature_set = scaler.transform(feature_set)

        return filename, Data(x=torch.tensor(feature_set, dtype=torch.float), edge_index=batch_edges, y=torch.tensor(target_set, dtype=torch.float))

    def __len__(self):
        return len(self.file_name)      


# To extract pose vectors from the openpose features
def _get_pose_vector(df,vec_def):
    A = df[[str(vec_def[0]+1).zfill(2)+'_x',str(vec_def[0]+1).zfill(2)+'_y']]
    B = df[[str(vec_def[1]+1).zfill(2)+'_x',str(vec_def[1]+1).zfill(2)+'_y']]
    vec = B.values.astype(float)-A.values.astype(float)
    # drop nan values
    vec = vec[np.isnan(vec).sum(axis=1)==0,:]
    vec = preprocessing.normalize(vec)
    return vec


# Dynamic + Openface and pose features
# Feature preprocessing done inside data loader only as it considers more than 1 frame  - no mean and meandiff features
class MPIIDatasetTemporalMain(data.Dataset):
    def __init__(self, file_info,feature_path_main, feature_path_cont,feature_pose_main, feature_pose_con, time, nodes):
        super(MPIIDatasetTemporalMain, self).__init__()
        info = pd.read_csv(file_info) 
        self.file_name = info['id']
        self.bc_label = info['label']
        self.num_nodes = nodes
        self.feature_path_main = feature_path_main
        self.feature_path_cont = feature_path_cont
        self.feature_pose_main = feature_pose_main
        self.feature_pose_con = feature_pose_con
        self.time = time

        # Features (columns) considered 
        au_cols = [' AU'+str(x).zfill(2)+'_r' for x in [1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45]]
        gaze_cols = [' gaze_0_x',' gaze_0_y',' gaze_0_z',' gaze_1_x',' gaze_1_y',' gaze_1_z']
        head_cols = [' pose_Tx',' pose_Ty',' pose_Tz',' pose_Rx',' pose_Ry',' pose_Rz']
        hgd_cols = ['nod_NS', 'shake_NS', 'tilt_NS']
        self.cols = au_cols + gaze_cols + head_cols +hgd_cols

        # pose angles to extract openpose features
        self.pose_angles = [[(5,6),(5,2)],[(2,3),(2,5)],[(3,4),(3,2)],
                        [(6,7),(6,5)],[(1,0),(1,5)],[(1,0),(1,2)],[(1,0),(1,12)],[(1,0),(1,9)],]
        # pose angles to extract openpose light features
        self.pose_angles_light = [[(5,6),(5,2)],[(2,3),(2,5)], 
                        [(3,4),(3,2)],[(1,0),(1,5)],[(1,0),(1,2)],[(1,0),(1,11)],[(1,0),(1,8)], ]

    def __getitem__(self, index):
        filename = self.file_name[index] #0_train_rec18_pos3
        main_pos = int(filename.split('_pos')[1])-1 # index of main 
        all_p = np.arange(self.num_nodes)
        con_id = np.setdiff1d(all_p, main_pos)

        # Main person
        feature_path = os.path.join(self.feature_path_main,self.file_name[index]+'_video.csv')
        df = pd.read_csv(feature_path, delimiter=',')
        df = df.iloc[-30:] # last 1 second
        df = df.fillna(0)
        feat_main = (df[self.cols]-df[self.cols].shift()).abs().fillna(0).values#.mean() # calculating diff
        
        # for pose main
        # feature_path_p = os.path.join(self.feature_pose_main,self.file_name[index]+'_video','pose_keypoints_2d.csv')
        # df = pd.read_csv(feature_path_p, delimiter=',')
        # df = df.iloc[-30:]
        # angle_features = []
        # for angle_id,pose_angle in enumerate(self.pose_angles):
        #     vec1 = _get_pose_vector(df,pose_angle[0])
        #     vec2 = _get_pose_vector(df,pose_angle[1])
        #     cur_angles = np.arccos(np.diag(np.dot(vec1,vec2.T)))
        #     if(len(cur_angles) !=30):
        #         left = 30-len(cur_angles)
        #         cur_angles = np.concatenate((cur_angles,cur_angles[-left:]),axis = 0)
        #     angle_features.append(cur_angles)
        #     # angle_features[:,angle_id] = cur_angles
        #     # angle_features['POSEANME_'+per_type+'_'+str(angle_id).zfill(2)] = cur_angles.mean()
        #     # angle_features['POSEANSD_'+per_type+'_'+str(angle_id).zfill(2)] = cur_angles.std()
        # del df
        # angle_features = np.asarray(angle_features).T
        # angle_features = np.diff(angle_features, axis  = 1)
        # print(filename,feat_main.shape, angle_features.shape)
        # feat_main = np.concatenate((feat_main,angle_features), axis = 1) #30,40


        # context persons
        flist = glob.glob(os.path.join(self.feature_path_cont,self.file_name[index].split('_pos')[0]+'*.csv'))
        flist.sort()
        feat_con = []
        angle_features = np.zeros([3,30,8])
        for c_id in range(len(flist)):
            df = pd.read_csv(flist[c_id], delimiter=',')
            df = df.iloc[-30:]
            df = df.fillna(0)
            ft = (df[self.cols]-df[self.cols].shift()).abs().fillna(0).values
            feat_con.append(ft)#df[self.cols].values)
            
            # feature_path_p = os.path.join(self.feature_pose_con,con_per+'_pose_keypoints.csv')
            # df = pd.read_csv(feature_path_p, delimiter=',')
            # df = df.iloc[-30:]
            
            # for angle_id,pose_angle in enumerate(self.pose_angles_light):
            #     vec1 = _get_pose_vector(df,pose_angle[0])
            #     vec2 = _get_pose_vector(df,pose_angle[1])
            #     cur_angles = np.arccos(np.diag(np.dot(vec1,vec2.T)))
            #     angle_features[c_id,:,angle_id] = cur_angles

        feat_con = np.asarray(feat_con)
        # angle_features = np.asarray(angle_features)
        # angle_features = np.diff(angle_features, axis  = 2)
        # feat_con = np.concatenate((feat_con,angle_features), axis = 2) #3,20,40

        target_set = []
        feature_set = None
        src = []
        dst = []
        all_main_nodes = []
        bc_label = self.bc_label[index]

        label_set = np.asarray([0, 0, 0,0]).astype(float)
        label_set[:] = bc_label
    

        for tc in range(self.time):
            # adding main person

            target_set.extend(label_set)
            if feature_set is None:
                feature_set = np.expand_dims(feat_main[tc,:], axis=0)
            else:
                feat = np.expand_dims(feat_main[tc,:], axis=0)
                feature_set = np.concatenate([feature_set, feat], axis=0)

            main_node_idx = feature_set.shape[0]-1
            all_main_nodes.append(main_node_idx)

            for ctx_entity in range(self.num_nodes - 1): # only for context
                feat = np.expand_dims(feat_con[ctx_entity, tc,:], axis=0)
                label = 0  # 0 for context persons
                feature_set = np.concatenate([feature_set, feat], axis=0)
                video_node_idx = feature_set.shape[0]-1

                src.extend([all_main_nodes[-1], video_node_idx, video_node_idx])
                dst.extend([video_node_idx, all_main_nodes[-1], video_node_idx])

        for i in range(len(feature_set)-self.num_nodes):
            src.append(i)
            dst.append(i+self.num_nodes)

        scaler = preprocessing.StandardScaler().fit(feature_set)
        feature_set = scaler.transform(feature_set)
        batch_edges = torch.tensor([src, dst], dtype=torch.long)

        return filename, Data(x=torch.tensor(feature_set, dtype=torch.float), edge_index=batch_edges, y=torch.tensor(target_set, dtype=torch.float))

    def __len__(self):
        return len(self.file_name)  


class MPIIDatasetAVFeats_Static(data.Dataset):
# For Static graph with extracted audio and visual deep features
# Input - 
# feature_path: path for Audio and video deep extracted features
# file_info: file containing the list of all the videos and their labels
# time: number of timestamps considered, 1 for static graph and num of frames used in case of dynamic 
# num_nodes: num of nodes in the graph
    def __init__(self, file_info,feature_path, num_nodes):
        super(MPIIDatasetAVFeats_Static, self).__init__()
        info = pd.read_csv(file_info) 
        self.file_name = info['id']
        self.bc_label = info['label']
        self.num_nodes = num_nodes
        self.feature_path = feature_path

    def __getitem__(self, index):
        filename = self.file_name[index] #0_train_rec18_pos3
        # print(filename)
        main_pos = int(filename.split('_pos')[1])-1 # index of main 
        all_p = np.arange(self.num_nodes)
        con_id = np.setdiff1d(all_p, main_pos)

        main_path = self.file_name[index].split('_pos')[0]+'_pos'+str(main_pos+1)+'_AudFt.npz'
        feature_path = os.path.join(self.feature_path,main_path)
        npzfile_1 = np.load(feature_path,allow_pickle = True)
        main_ft = npzfile_1['arr_0']
        main_ft_aud = main_ft[-30:]
        main_ft_aud = np.mean(main_ft_aud, axis = 0)

        main_path = self.file_name[index].split('_pos')[0]+'_pos'+str(main_pos+1)+'_VidFt.npz'
        feature_path = os.path.join(self.feature_path,main_path)
        npzfile_1 = np.load(feature_path,allow_pickle = True)
        main_ft = npzfile_1['arr_0']
        main_ft_vid = main_ft[-30:]
        main_ft_vid = np.mean(main_ft_vid, axis = 0)

        con1_path = self.file_name[index].split('_pos')[0]+'_pos'+str(con_id[0]+1)+'_AudFt.npz'
        feature_path = os.path.join(self.feature_path,con1_path)
        npzfile_1 = np.load(feature_path,allow_pickle = True)
        con1_ft = npzfile_1['arr_0']
        con1_ft_aud = con1_ft[-30:]
        con1_ft_aud = np.mean(con1_ft_aud, axis = 0)
        
        con1_path = self.file_name[index].split('_pos')[0]+'_pos'+str(con_id[0]+1)+'_VidFt.npz'
        feature_path = os.path.join(self.feature_path,con1_path)
        npzfile_1 = np.load(feature_path,allow_pickle = True)
        con1_ft = npzfile_1['arr_0']
        con1_ft_vid = con1_ft[-30:]
        con1_ft_vid = np.mean(con1_ft_vid, axis = 0)
        
        con2_path = self.file_name[index].split('_pos')[0]+'_pos'+str(con_id[1]+1)+'_AudFt.npz'
        feature_path = os.path.join(self.feature_path,con2_path)
        npzfile_1 = np.load(feature_path,allow_pickle = True)
        con2_ft = npzfile_1['arr_0']
        con2_ft_aud = con2_ft[-30:]
        con2_ft_aud = np.mean(con2_ft_aud, axis = 0)

        con2_path = self.file_name[index].split('_pos')[0]+'_pos'+str(con_id[1]+1)+'_VidFt.npz'
        feature_path = os.path.join(self.feature_path,con2_path)
        npzfile_1 = np.load(feature_path,allow_pickle = True)
        con2_ft = npzfile_1['arr_0']
        con2_ft_vid = con2_ft[-30:]
        con2_ft_vid = np.mean(con2_ft_vid, axis = 0)
        
        con3_path = self.file_name[index].split('_pos')[0]+'_pos'+str(con_id[2]+1)+'_AudFt.npz'
        feature_path = os.path.join(self.feature_path,con3_path)
        npzfile_1 = np.load(feature_path,allow_pickle = True)
        con3_ft = npzfile_1['arr_0']
        con3_ft_aud = con3_ft[-30:]
        con3_ft_aud = np.mean(con3_ft_aud, axis = 0)

        con3_path = self.file_name[index].split('_pos')[0]+'_pos'+str(con_id[2]+1)+'_VidFt.npz'
        feature_path = os.path.join(self.feature_path,con3_path)
        npzfile_1 = np.load(feature_path,allow_pickle = True)
        con3_ft = npzfile_1['arr_0']
        con3_ft_vid = con3_ft[-30:]
        con3_ft_vid = np.mean(con3_ft_vid, axis = 0)

        main_ft_aud = np.expand_dims(main_ft_aud, axis=0)
        con1_ft_aud = np.expand_dims(con1_ft_aud, axis=0)
        con2_ft_aud = np.expand_dims(con2_ft_aud, axis=0)
        con3_ft_aud = np.expand_dims(con3_ft_aud, axis=0)

        main_ft_vid = np.expand_dims(main_ft_vid, axis=0)
        con1_ft_vid = np.expand_dims(con1_ft_vid, axis=0)
        con2_ft_vid = np.expand_dims(con2_ft_vid, axis=0)
        con3_ft_vid = np.expand_dims(con3_ft_vid, axis=0)

        feature_set = np.concatenate([main_ft_aud,con1_ft_aud,con2_ft_aud,con3_ft_aud, main_ft_vid,con1_ft_vid,con2_ft_vid, con3_ft_vid], axis = 0) #4, 1024
        # print(feature_set.shape)
        src = []
        dst = []
        # bidirectional connextions between main and context person only
        src.extend([0,0,0,0,1,2,3,4,4,4,4,5,6,7,0,0,0,5,6,7,4,4,4,1,2,3])
        dst.extend([0,1,2,3,0,0,0,4,5,6,7,4,4,4,5,6,7,0,0,0,1,2,3,4,4,4])

        batch_edges = torch.tensor([src, dst], dtype=torch.long)
        bc_label = self.bc_label[index]

        target_set = np.asarray([0,0,0,0,0,0,0,0]).astype(float)
        target_set[:] = bc_label

        scaler = preprocessing.StandardScaler().fit(feature_set)
        feature_set = scaler.transform(feature_set)

        batch_edges = torch.tensor([src, dst], dtype=torch.long)

        return filename, Data(x=torch.tensor(feature_set, dtype=torch.float), edge_index=batch_edges, y=torch.tensor(target_set, dtype=torch.float))

    def __len__(self):
        return len(self.file_name)

class MPIIDatasetAVFeats_Dynamic(data.Dataset):
# For Dynamic graph with extracted audio and visual deep features
# Input - 
# feature_path: path for Audio and video deep extracted features
# file_info: file containing the list of all the videos and their labels
# time: number of timestamps considered, 1 for static graph and num of frames used in case of dynamic 
# num_nodes: num of nodes in the graph

    def __init__(self, file_info,feature_path, time, nodes):
        super(MPIIDatasetAVFeats_Dynamic, self).__init__()
        info = pd.read_csv(file_info) 
        self.file_name = info['id']
        self.bc_label = info['label']
        self.num_nodes = nodes
        self.feature_path = feature_path
        self.time = time

    def __getitem__(self, index):
        filename = self.file_name[index] #0_train_rec18_pos3
        # print(filename)
        main_pos = int(filename.split('_pos')[1])-1 # index of main
        all_p = np.arange(self.num_nodes)
        con_id = np.setdiff1d(all_p, main_pos)

        # Taking the audio and video extracted features corresponding to the main person
        main_path = self.file_name[index].split('_pos')[0]+'_pos'+str(main_pos+1)+'_AudFt.npz'
        feature_path = os.path.join(self.feature_path,main_path)
        npzfile_1 = np.load(feature_path,allow_pickle = True)
        main_ft = npzfile_1['arr_0']
        main_ft_aud = main_ft[-30:] # only for last 1 sec - Why? Refer the paper

        main_path = self.file_name[index].split('_pos')[0]+'_pos'+str(main_pos+1)+'_VidFt.npz'
        feature_path = os.path.join(self.feature_path,main_path)
        npzfile_1 = np.load(feature_path,allow_pickle = True)
        main_ft = npzfile_1['arr_0']
        main_ft_vid = main_ft[-30:]

        # For context features
        con1_path = self.file_name[index].split('_pos')[0]+'_pos'+str(con_id[0]+1)+'_AudFt.npz'
        feature_path = os.path.join(self.feature_path,con1_path)
        npzfile_1 = np.load(feature_path,allow_pickle = True)
        con1_ft = npzfile_1['arr_0']
        con1_ft_aud = con1_ft[-30:]

        con1_path = self.file_name[index].split('_pos')[0]+'_pos'+str(con_id[0]+1)+'_VidFt.npz'
        feature_path = os.path.join(self.feature_path,con1_path)
        npzfile_1 = np.load(feature_path,allow_pickle = True)
        con1_ft = npzfile_1['arr_0']
        con1_ft_vid = con1_ft[-30:]
        
        con2_path = self.file_name[index].split('_pos')[0]+'_pos'+str(con_id[1]+1)+'_AudFt.npz'
        feature_path = os.path.join(self.feature_path,con2_path)
        npzfile_1 = np.load(feature_path,allow_pickle = True)
        con2_ft = npzfile_1['arr_0']
        con2_ft_aud = con2_ft[-30:]

        con2_path = self.file_name[index].split('_pos')[0]+'_pos'+str(con_id[1]+1)+'_VidFt.npz'
        feature_path = os.path.join(self.feature_path,con2_path)
        npzfile_1 = np.load(feature_path,allow_pickle = True)
        con2_ft = npzfile_1['arr_0']
        con2_ft_vid = con2_ft[-30:]
        
        con3_path = self.file_name[index].split('_pos')[0]+'_pos'+str(con_id[2]+1)+'_AudFt.npz'
        feature_path = os.path.join(self.feature_path,con3_path)
        npzfile_1 = np.load(feature_path,allow_pickle = True)
        con3_ft = npzfile_1['arr_0']
        con3_ft_aud = con3_ft[-30:]

        con3_path = self.file_name[index].split('_pos')[0]+'_pos'+str(con_id[2]+1)+'_VidFt.npz'
        feature_path = os.path.join(self.feature_path,con3_path)
        npzfile_1 = np.load(feature_path,allow_pickle = True)
        con3_ft = npzfile_1['arr_0']
        con3_ft_vid = con3_ft[-30:]

        main_ft_aud = np.expand_dims(main_ft_aud, axis=0)
        con1_ft_aud = np.expand_dims(con1_ft_aud, axis=0)
        con2_ft_aud = np.expand_dims(con2_ft_aud, axis=0)
        con3_ft_aud = np.expand_dims(con3_ft_aud, axis=0)
        con_ft_aud = np.concatenate((con1_ft_aud, con1_ft_aud, con1_ft_aud), axis = 0)
        # print(con_ft_aud.shape)

        main_ft_vid = np.expand_dims(main_ft_vid, axis=0)
        con1_ft_vid = np.expand_dims(con1_ft_vid, axis=0)
        con2_ft_vid = np.expand_dims(con2_ft_vid, axis=0)
        con3_ft_vid = np.expand_dims(con3_ft_vid, axis=0)
        con_ft_vid = np.concatenate((con1_ft_vid, con1_ft_vid, con1_ft_vid), axis = 0)

        target_set = []
        feature_set = None
        src = []
        dst = []
        all_main_nodes = []
        bc_label = self.bc_label[index]

        label_set = np.asarray([0,0,0,0,0,0,0,0]).astype(float)
        label_set[:] = bc_label #considering same label for all nodes

        # Temporally Edge matrix and features
        # At 1 time t all nodes connected to each other. 
        # At other t each node connected to each respective nodes
        for tc in range(self.time):
            # adding main person
            target_set.extend(label_set)
            if feature_set is None:
                feature_set = np.expand_dims(main_ft_aud[0,tc,:], axis=0)
            else:
                feat = np.expand_dims(main_ft_aud[0,tc,:], axis=0)
                feature_set = np.concatenate([feature_set, feat], axis=0)

            main_node_idx = feature_set.shape[0]-1
            all_main_nodes.append(main_node_idx)
            # print('feature_set',feature_set.shape)

            for ctx_entity in range(self.num_nodes - 1): # only for context
                label = 0  # 0 for context persons
                feature_set = np.concatenate([feature_set, feat], axis=0)
                video_node_idx = feature_set.shape[0]-1
                src.extend([all_main_nodes[-1], video_node_idx, video_node_idx])
                dst.extend([video_node_idx, all_main_nodes[-1], video_node_idx])
            # video
            feat = np.expand_dims(main_ft_vid[0,tc,:], axis=0)
            feature_set = np.concatenate([feature_set, feat], axis=0)
            main_node_idx = feature_set.shape[0]-1
            all_main_nodes.append(main_node_idx)
            for ctx_entity in range(self.num_nodes - 1): # only for context
                feat = np.expand_dims(con_ft_vid[ctx_entity, tc,:], axis=0)
                label = 0  # 0 for context persons
                feature_set = np.concatenate([feature_set, feat], axis=0)
                video_node_idx = feature_set.shape[0]-1
                src.extend([all_main_nodes[-1], video_node_idx, video_node_idx])
                dst.extend([video_node_idx, all_main_nodes[-1], video_node_idx])

        for i in range(len(feature_set)-self.num_nodes):
            src.append(i)
            dst.append(i+self.num_nodes)

        scaler = preprocessing.StandardScaler().fit(feature_set)
        feature_set = scaler.transform(feature_set)
        batch_edges = torch.tensor([src, dst], dtype=torch.long)

        return filename, Data(x=torch.tensor(feature_set, dtype=torch.float), edge_index=batch_edges, y=torch.tensor(target_set, dtype=torch.float))

    def __len__(self):
        return len(self.file_name)  
