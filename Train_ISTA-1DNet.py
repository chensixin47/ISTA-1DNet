#Import library
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader


#%%
#Directory of the training
Channel='_A5'
Layer_number='_18'
Dictionary_size='_8192'
Gamma='_0.01'
Learning_rate='_1e-5'
Dictionary_initialization='_DCT'
Model_dir = 'model'+Channel+Layer_number+Dictionary_size+Gamma+Learning_rate+Dictionary_initialization
Log_file_name = 'log'+Channel+Layer_number+Dictionary_size+Gamma+Learning_rate+Dictionary_initialization


#%%
'''7 important hyperparameters
1 cs_ratio
2 start_epoch
3 end_epoch
4 layer_num
5 learning_rate
6 batch_size
7 dictionary_size
8 regularization_Gamma
'''
layer_dict = {'_9':9,'_12':12,'_15':15,'_18':18, '_21':21,'_24':24}
cs_ratio = '50'
start_epoch = 0
end_epoch = 5
layer_num = layer_dict[Layer_number]
learning_rate = 1e-5
batch_size = 32
if Dictionary_size == '_8192':
    dictionary_size = 8192 
if Dictionary_size == '_4096':
    dictionary_size = 4096
if Gamma == '_0.1':
    regularization_Gamma = 0.1 #Gamma='_0.1'
if Gamma == '_0.01':
    regularization_Gamma = 0.01


#%%
#Set hardware
group_num = 1
gpu_list = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%%
#Load training data
Training_data = sio.loadmat('data/A5.mat') #Channel='_A5'
training_set_A5 = Training_data['training_set_A5'].T
Training_labels = training_set_A5
print('Training labels shape:',Training_labels.shape)
nrtrain = Training_labels.shape[0]


#%%
#Load measurement matrix
Measurement_matrix = sio.loadmat('sampling_matrix/RD.mat')
Phi_input = Measurement_matrix[cs_ratio]
print('Phi shape:',Phi_input.shape)


#%%
# Compute Initialization Matrix:
Qinit_Name = './sampling_matrix/Initialization_Matrix_%s.mat' % (cs_ratio)
if os.path.exists(Qinit_Name):
    Qinit_data = sio.loadmat(Qinit_Name)
    Qinit = Qinit_data['Qinit']
    print('Qinit already exists, shape:',Qinit.shape)
else:
    X_data = Training_labels.transpose()
    Y_data = np.dot(Phi_input, X_data)
    Y_YT = np.dot(Y_data, Y_data.transpose())
    X_YT = np.dot(X_data, Y_data.transpose())
    Qinit = np.dot(X_YT, np.linalg.inv(Y_YT))
    del X_data, Y_data, X_YT, Y_YT
    sio.savemat(Qinit_Name, {'Qinit': Qinit})

Dictionary_Name = './sampling_matrix/Dictionary_Matirx.mat'
if dictionary_size == 8192:
    print('Redundant')
    Dictionary = sio.loadmat(Dictionary_Name)['DCT2DCT4BASIS']


#%%
# Define ISTA-Net Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        if Dictionary_initialization=='_No':
            self.transform_matrix = nn.Parameter(init.xavier_normal_(torch.Tensor(dictionary_size,4096)))#Dictionary_initialization='_No'
            self.inverse_transform_matrix = nn.Parameter(init.xavier_normal_(torch.Tensor(4096,dictionary_size)))                            
        if Dictionary_initialization=='_DCT':
            self.transform_matrix = nn.Parameter(torch.tensor(Dictionary.transpose(),dtype=torch.float))#Dictionary_initialization='_DCT'
            self.inverse_transform_matrix = nn.Parameter(torch.tensor(Dictionary,dtype=torch.float))

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        # print('r shape:',x.shape)
        x_input = x
        x_forward = F.linear(x_input,self.transform_matrix)
        # print('F(r) shape:',x_forward.shape)
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))
        # print('Soft(F(r)) shape', x.shape)
        x_backward = F.linear(x,self.inverse_transform_matrix)
        # print('F_tilde（Soft(F(r))) shape：', x_backward.shape)
        x_pred = x_backward
        x_est = F.linear(x_forward,self.inverse_transform_matrix)
        # print('x_est shape:',x_est.shape)
        symloss = x_est - x_input
        return [x_pred, symloss]


#%%
# Define ISTA-Net
class ISTANet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTANet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        for i in range(1):
            onelayer.append(BasicBlock())
        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit):
        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)
        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))
        layers_sym = []   # for computing symmetric loss
        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[0](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)
        x_final = x
        return [x_final, layers_sym]


#%%
#Instantiate ISTA-1DNet
model = ISTANet(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

print_flag = 1   # print parameter number
if print_flag:
    num_count = 0
    for para in model.parameters():
        num_count += 1
        print('Layer %d' % num_count)
        print(para.size())

if not os.path.exists(Model_dir):
    os.makedirs(Model_dir)

if start_epoch > 0:
    print('Model already exists! Use pretrained parameters.')
    pre_model_dir = Model_dir
    model.load_state_dict(torch.load('./%s/net_params_optimizer_%d.pkl' % (pre_model_dir, start_epoch-1)))


#%%
#Define Dataloader and optimizer
class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

Phi = torch.from_numpy(Phi_input).type(torch.FloatTensor)
Phi = Phi.to(device)

Qinit = torch.from_numpy(Qinit).type(torch.FloatTensor)
Qinit = Qinit.to(device)


#%%
# Prerun the ISTA-1DNet
data = next(iter(rand_loader))
batch_x = data
# print('batch_x shape:',batch_x.shape)
batch_x = batch_x.to(device)
Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))
# print('Phix shape:',Phix.shape)
[x_output, loss_layers_sym] = model(Phix, Phi, Qinit)
print('x_output shape:',x_output.shape)


#%%
# Training loop
for epoch_i in range(start_epoch, end_epoch):
    for ii, data in enumerate(rand_loader):

        batch_x = data
        batch_x = batch_x.to(device)

        Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))

        [x_output, loss_layers_sym] = model(Phix, Phi, Qinit)

        # Compute and print loss
        loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))

        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        for k in range(layer_num-1):
            loss_constraint += torch.mean(torch.pow(loss_layers_sym[k+1], 2))

        gamma = torch.Tensor([regularization_Gamma]).to(device)

        # loss_all = loss_discrepancy
        loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        if ii % 10 == 9:
            output_data = "[%02d/%02d:%2d] Total Loss: %.4f, Discrepancy Loss: %.4f,  Constraint Loss: %.4f\n" % (epoch_i, end_epoch, ii, loss_all.item(), loss_discrepancy.item(), loss_constraint)
            print(output_data)

    output_file = open(Log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if epoch_i % 10 == 9:
        torch.save(model.state_dict(), "./%s/net_params_optimizer_%d.pkl" % (Model_dir, epoch_i))
