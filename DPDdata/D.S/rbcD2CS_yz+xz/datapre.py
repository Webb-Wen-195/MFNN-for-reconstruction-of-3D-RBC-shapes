import numpy as np

np.random.seed(1117)

class DataSet:
    def __init__(self, batch_size_lf):

        self.batch_size = batch_size_lf
        
        data_lf = np.load('./data/D_1_lf.npz')
        sphere_x_lf = data_lf['sphere']
        rbc_u_lf = data_lf['rbc']
        self.tri_archive = data_lf['triangles']
        self.AV_lf = data_lf['AV'] # AV of base RBC

        data_hf_all = np.load('./data/S_mu80_baseD_hf.npz')
        sphere_x_hf_all = data_hf_all['sphere']
        self.AV_hf = data_hf_all['AV'] # AV of predicted RBC

        data_hf = np.load('./data/S_mu80_baseD_hf_partial_xy.npz')
        sphere_x_hf_train = data_hf['sphere_p']
        rbc_u_hf_train = data_hf['rbc_p']
        sphere_x_hf_test = data_hf['sphere_r']
        rbc_u_hf_test = data_hf['rbc_r']


        num_lf_train = 2300

        indices = np.arange(rbc_u_lf.shape[0])
        np.random.shuffle(indices)
        train_indices = indices[:num_lf_train ]
        test_indices = indices[num_lf_train :]

        # x -> initial sphere coord set, u -> rbc coord set
        x_lf_train = sphere_x_lf[train_indices, 1:]
        x_lf_test = sphere_x_lf[test_indices, 1:]
        u_lf_train = rbc_u_lf[train_indices, 1:]
        u_lf_test = rbc_u_lf[test_indices, 1:]

        x_hf_train = sphere_x_hf_train[:, 1:]
        u_hf_train = rbc_u_hf_train[:, 1:]
        x_hf_test = sphere_x_hf_test[:, 1:]
        u_hf_test = rbc_u_hf_test[:, 1:]
        
        x_mean = np.mean(x_lf_train, axis=0, keepdims=True)
        x_std = np.std(x_lf_train, axis=0, keepdims=True)
        u_mean = np.mean(u_lf_train, axis=0, keepdims=True)
        u_std = np.std(u_lf_train, axis=0, keepdims=True)

        x_lf_train = (x_lf_train - x_mean)/(x_std + 1.0e-6)
        u_lf_train = (u_lf_train - u_mean)/(u_std + 1.0e-6) 
        x_lf_test = (x_lf_test - x_mean)/(x_std + 1.0e-6)
        x_hf_train = (x_hf_train - x_mean)/(x_std + 1.0e-6)
        u_hf_train = (u_hf_train - u_mean)/(u_std + 1.0e-6)
        x_hf_test = (x_hf_test - x_mean)/(x_std + 1.0e-6)

        self.u_mean = u_mean
        self.u_std = u_std

        self.x_lf_train = x_lf_train
        self.u_lf_train = u_lf_train
        self.x_lf_test = x_lf_test
        self.u_lf_test = u_lf_test
        self.x_hf_train = x_hf_train
        self.u_hf_train = u_hf_train
        self.x_hf_test = x_hf_test
        self.u_hf_test = u_hf_test

        self.x_lf_all = np.hstack((sphere_x_lf[:, :1],(sphere_x_lf[:,1:] - x_mean)/(x_std + 1.0e-6)))
        self.x_hf_all = np.hstack((sphere_x_hf_all[:, :1],(sphere_x_hf_all[:,1:] - x_mean)/(x_std + 1.0e-6)))

    def decode(self, x):
        return x*(self.u_std  + 1.0e-6) + self.u_mean
    
    def minibatch(self):    
        batch_id = np.random.choice(self.x_lf_train.shape[0], self.batch_size, replace=False)
        x_train_lf_batch = self.x_lf_train[batch_id]
        u_train_lf_batch = self.u_lf_train[batch_id]
        return x_train_lf_batch, u_train_lf_batch
        

