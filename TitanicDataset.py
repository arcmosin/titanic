import torch
from torch.utils.data import Dataset
import numpy as np
import csv

class TitanicDataset(Dataset):
    def __init__(self,path,mode='train'):
        self.mode=mode
        data=self.open_path(path)
        self.x_data=torch.from_numpy(self.x_data_deal(data))
        if self.mode == 'train':
            self.y_data=torch.from_numpy(self.y_data_deal(data))
        self.len = data.shape[0]

    def __getitem__(self,index):
        if self.mode=='train':
            return self.x_data[index],self.y_data[index]
        else:
            return self.x_data[index]

    def __len__(self):
        return self.len

    def completAge(self,data):
        age_mr = []
        age_mrs = []
        age_ms = []
        age_miss = []
        age_other = []
        for j in range(data.shape[0]):
            if ('Mr.' in data[j][2]) & (data[j][4] != ''):
                age_mr.append(int(float(data[j][4])))
            elif ('Mrs.' in data[j][2]) & (data[j][4] != ''):
                age_mrs.append(int(float(data[j][4])))
            elif ('Ms.' in data[j][2]) & (data[j][4] != ''):
                age_ms.append(int(float(data[j][4])))
            elif ('Miss.' in data[j][2]) & (data[j][4] != ''):
                age_miss.append(int(float(data[j][4])))
            elif (data[j][4] != ''):
                age_other.append(int(float(data[j][4])))
        mean_mr = np.mean(age_mr)
        mean_mrs = np.mean(age_mrs)
        mean_ms = np.mean(age_ms)
        mean_miss = np.mean(age_miss)
        mean_other = np.mean(age_other)

        for k in range(data.shape[0]):
            if ('Mr.' in data[j][2]) & (data[j][4] == ''):
                data[k][4] = str(mean_mr)
            elif ('Mrs.' in data[j][2]) & (data[j][4] == ''):
                data[k][4] = str(mean_mrs)
            elif ('Ms.' in data[j][2]) & (data[j][4] == ''):
                data[k][4] = str(mean_ms)
            elif ('Miss.' in data[j][2]) & (data[j][4] == ''):
                data[k][4] = str(mean_miss)
            elif data[k][4] == '':
                data[k][4] = str(mean_other)
            else:
                pass
        return data

    def open_path(self,path):
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])
            data = data[:, 1:]
            if self.mode=='test':
                data = np.c_[data[:, 0], data[:,:]]
            data = self.completAge(data)
        return data

    def x_data_deal(self,data):
        x_data = np.c_[data[:, 1], data[:, 3], data[:, 4]]
        for i in range(x_data.shape[0]):
            if x_data[i][1] == 'male':
                x_data[i][1] = 1
            else:
                x_data[i][1] = 0
        x_data = x_data.astype(np.float32)
        return x_data

    def y_data_deal(self, data):
        y_data = data[:, 0]
        y_data = y_data.astype(np.float32)
        return y_data