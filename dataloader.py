import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import scipy.io
import torch
import h5py
class CCV(Dataset):#6773*5000
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773


    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class Hdigit():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Hdigit.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(10000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][0][1].T.astype(np.float32)
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class BBC(Dataset):
    def __init__(self, name):
        data_path = './data/{}.mat'.format(name[1])
        np.random.seed(1)
        index = [i for i in range(name['N'])]
        np.random.shuffle(index)

        data = h5py.File(data_path)
        Final_data = []
        for i in range(name['V']):
            diff_view = data[data['X'][0, i]]
            diff_view = np.array(diff_view, dtype=np.float32).T
            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        label = np.array(data['Y']).T
        LABELS = label[index]
        self.name = name
        self.data = Final_data
        self.y = LABELS

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if self.name['V'] == 2:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx])], torch.from_numpy(
                self.y[idx]), torch.from_numpy(np.array(idx)).long()
        elif self.name['V'] == 3:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx]),
                    torch.from_numpy(self.data[2][idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(
                np.array(idx)).long()
        elif self.name['V'] == 4:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx]),
                    torch.from_numpy(self.data[2][idx]), torch.from_numpy(self.data[3][idx])], \
                   torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
        elif self.name['V'] == 5:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx]),
                    torch.from_numpy(self.data[2][idx]), torch.from_numpy(self.data[3][idx]),
                    torch.from_numpy(self.data[4][idx])], \
                   torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
        elif self.name['V'] == 6:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx]),
                    torch.from_numpy(self.data[2][idx]), torch.from_numpy(self.data[3][idx]),
                    torch.from_numpy(self.data[4][idx]), torch.from_numpy(self.data[5][idx])], \
                   torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
        elif self.name['V'] == 7:
            return [torch.from_numpy(self.data[0][idx]), torch.from_numpy(self.data[1][idx]),
                    torch.from_numpy(self.data[2][idx]), torch.from_numpy(self.data[3][idx]),
                    torch.from_numpy(self.data[4][idx]), torch.from_numpy(self.data[5][idx]),
                    torch.from_numpy(self.data[6][idx])], \
                   torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
        else:
            raise NotImplementedError


data_info = dict(
    BBCSport = {1: 'BBCSport', 'N': 544, 'K': 5, 'V': 2, 'n_input': [3183,3203], 'n_hid': [512,512], 'n_output': 64},
)
def load_data(dataset):
    if dataset == "Hdigit":
        dataset = Hdigit('./data/')
        dims = [784, 256]
        view = 2
        data_size = 10000
        class_num = 10
    elif dataset =="BBCSport":
        dataset_para = data_info[dataset]
        dataset = BBC(dataset_para)
        dims = dataset_para['n_input']
        view = dataset_para['V']
        data_size = dataset_para['N']
        class_num = dataset_para['K']
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
