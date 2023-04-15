import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp

def data_load(dataset, has_v=True, has_a=True, has_t=True):
    dir_str = './Data/' + dataset

    if dataset == 'movielens':
        user_num = 55485
        item_num = 5986
        train_edge = np.load(dir_str + '/train.npy', allow_pickle=True)
        user_item_dict = np.load(dir_str + '/user_item_dict.npy', allow_pickle=True).item()
        v_feat = torch.tensor(np.load(dir_str + '/FeatureVideo_normal.npy', allow_pickle=True), dtype=torch.float).cuda() if has_v else None
        a_feat = torch.tensor(np.load(dir_str + '/FeatureAudio_avg_normal.npy', allow_pickle=True), dtype=torch.float).cuda() if has_a else None
        t_feat = torch.tensor(np.load(dir_str + '/FeatureText_stl_normal.npy', allow_pickle=True), dtype=torch.float).cuda() if has_t else None
    elif dataset == 'tiktok':
        user_num = 36656
        item_num = 76085
        train_edge = np.load(dir_str + '/train.npy', allow_pickle=True)
        user_item_dict = np.load(dir_str + '/user_item_dict.npy', allow_pickle=True).item()
        v_feat = torch.load(dir_str + '/visual_feat_new.pt').to(dtype=torch.float).cuda() if has_v else None
        a_feat = torch.load(dir_str + '/audio_feat_new.pt').to(dtype=torch.float).cuda() if has_a else None
        t_feat = torch.tensor(np.load(dir_str + '/tiktok_t_64.npy')).to(dtype=torch.float).cuda() if has_t else None
    elif dataset == 'kwai':
        user_num = 7010
        item_num = 86483
        train_edge = np.load(dir_str + '/train.npy', allow_pickle=True)
        user_item_dict = np.load(dir_str + '/user_item_dict.npy', allow_pickle=True).item()
        v_feat = torch.load(dir_str + '/v_feat.pt').to(dtype=torch.float).cuda() if has_v else None
        a_feat = None
        t_feat = torch.tensor(np.load(dir_str + '/kwai_t_64.npy')).to(dtype=torch.float).cuda() if has_t else None

    train_edge[:, 1] += user_num
    user_item_dict = {i:[j+user_num for j in user_item_dict[i]] for i in user_item_dict.keys()}

    return user_num, item_num, train_edge, user_item_dict, v_feat, a_feat, t_feat

class TrainingDataset(Dataset):
    def __init__(self, dataset, user_num, item_num, user_item_dict, edge_index, src_len):
        self.dir_str = './Data/' + dataset
        self.user_num = user_num
        self.item_num = item_num
        self.user_item_dict = user_item_dict
        self.edge_index = edge_index
        self.src_len = src_len
        self.all_set = set(range(user_num, user_num + item_num))
        self.graph = None

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, index):
        user, pos_item = self.edge_index[index]
        while True:
            neg_item = random.sample(self.all_set, 1)[0]
            if neg_item not in self.user_item_dict[user]:
                break

        temp = list(self.user_item_dict[user])
        random.shuffle(temp)
        if len(temp) > self.src_len:
            mask = torch.ones(self.src_len + 1) == 0
            temp = temp[:self.src_len]
        else:
            mask = torch.cat((torch.ones(len(temp) + 1), torch.zeros(self.src_len - len(temp)))) == 0
            temp.extend([self.user_num for i in range(self.src_len - len(temp))])

        user_item = torch.tensor(temp) - self.user_num
        user_item = torch.cat((torch.tensor([-1]), user_item))

        return torch.LongTensor([user,user]), torch.LongTensor([pos_item, neg_item]), user_item, mask

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_sparse_graph(self):
        # print('loading adjacency matrix')
        if self.graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.dir_str + '/s_pre_adj_mat.npz')
                # print('successfully loaded...')
                norm_adj = pre_adj_mat
            except:
                # print('generating adjacency matrix')
                adj_mat = sp.dok_matrix((self.user_num + self.item_num, self.user_num + self.item_num), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                train_user = self.edge_index[:, 0]
                train_item = self.edge_index[:, 1] - self.user_num
                R = csr_matrix((np.ones(len(train_user)), (train_user, train_item)), shape=(self.user_num, self.item_num)).tolil()
                adj_mat[:self.user_num, self.user_num:] = R
                adj_mat[self.user_num:, :self.user_num] = R.T
                adj_mat = adj_mat.todok()
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum + 1e-5, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                
                sp.save_npz(self.dir_str + '/s_pre_adj_mat.npz', norm_adj)
            
            self.graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.graph = self.graph.coalesce().cuda()

        return self.graph

class EvalDataset(Dataset):
    def __init__(self, dataset, user_num, item_num, user_item_dict, src_len):
        self.dir_str = './Data/' + dataset
        self.user_num = user_num
        self.item_num = item_num
        self.user_item_dict = user_item_dict
        self.src_len = src_len

    def __len__(self):
        return self.user_num

    def __getitem__(self, index):
        user = index

        temp = list(self.user_item_dict[user])
        random.shuffle(temp)
        if len(temp) > self.src_len:
            mask = torch.ones(self.src_len + 1) == 0
            temp = temp[:self.src_len]
        else:
            mask = torch.cat((torch.ones(len(temp) + 1), torch.zeros(self.src_len - len(temp)))) == 0
            temp.extend([self.user_num for i in range(self.src_len - len(temp))])

        user_item = torch.tensor(temp) - self.user_num
        user_item = torch.cat((torch.tensor([-1]), user_item))
        
        return torch.LongTensor([user]), user_item, mask
