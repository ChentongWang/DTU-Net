import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.utils.checkpoint as checkpoint
import os
from model import multiStageUnmixing

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Random Seed
seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load DATA
# Notably, your own dataset should include:
# Y:  L*N -------- L bands, N pixels
# M:  L*R -------- R ground truth endmembers with L bands
# A:  R*N -------- Ground truth abundance maps
# M1: L*R -------- Endmembers by VCA (or other initializations)
# b:  1*N -------- Ground truth nonlinear coefficients (for PPNMM-generated synthetic datasets)
data = sio.loadmat("Name_of_Your_Data.mat")

abundance_GT = torch.from_numpy(data["A"])  # true abundance
original_HSI = torch.from_numpy(data["Y"])  # HSI data

# For PPNMM-generated synthetic datasets, load the true nonlinear coefficients b. If not, just comment out it.
b_true = torch.from_numpy(data["b"])

# VCA_endmember and GT
VCA_endmember = data["M1"]
GT_endmember = data["M"]
endmember_init = torch.from_numpy(VCA_endmember).unsqueeze(2).unsqueeze(3).float()
GT_init = torch.from_numpy(GT_endmember).unsqueeze(2).unsqueeze(3).float()

band_Number = original_HSI.shape[0]
endmember_number, pixel_number = abundance_GT.shape
col = 100

# observed image and GT abundance
original_HSI = torch.reshape(original_HSI, (band_Number, col, col))
abundance_GT = torch.reshape(abundance_GT, (endmember_number, col, col))

batch_size = 1
EPOCH = 600
alpha = 0.1
beta = 0.03
drop_out = 0.
learning_rate = 0.01


class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(1e-6, 1)

# abundance normalization
def norm_abundance_GT(abundance_input, abundance_GT_input):
    abundance_input = abundance_input / (torch.sum(abundance_input, dim=1))
    abundance_input = torch.reshape(
        abundance_input.squeeze(0), (endmember_number, col, col)
    )
    abundance_input = abundance_input.cpu().detach().numpy()
    abundance_GT_input = abundance_GT_input / (torch.sum(abundance_GT_input, dim=0))
    abundance_GT_input = abundance_GT_input.cpu().detach().numpy()
    return abundance_input, abundance_GT_input


# endmember normalization
def norm_endmember(endmember_input, endmember_GT):
    for i in range(0, endmember_number):
        endmember_input[:, i] = endmember_input[:, i] / np.max(endmember_input[:, i])
        endmember_GT[:, i] = endmember_GT[:, i] / np.max(endmember_GT[:, i])
    return endmember_input, endmember_GT


# plot abundance
def plot_abundance(abundance_input, abundance_GT_input):
    plt.figure(figsize=(60, 25))
    for i in range(0, endmember_number):

        plt.subplot(2, endmember_number, i + 1)
        plt.pcolor(abundance_input[i, :, :], cmap='jet')
        plt.colorbar(shrink=.83)

        plt.subplot(2, endmember_number, endmember_number + i + 1)
        plt.pcolor(abundance_GT_input[i, :, :], cmap='jet')
        plt.colorbar(shrink=.83)
    plt.show()


# plot endmember
def plot_endmember(endmember_input, endmember_GT):
    plt.figure(figsize=(13, 2.5), dpi=150)
    for i in range(0, endmember_number):
        plt.subplot(1, endmember_number, i + 1)
        plt.plot(endmember_input[:, i], label="Extracted")
        plt.plot(endmember_GT[:, i], label="GT")
    plt.legend()
    plt.show()


# change the index of abundance and endmember
def arange_A_E(abundance_input, abundance_GT_input, endmember_input, endmember_GT):
    RMSE_matrix = np.zeros((endmember_number, endmember_number))
    SAD_matrix = np.zeros((endmember_number, endmember_number))
    RMSE_index = np.zeros(endmember_number).astype(int)
    SAD_index = np.zeros(endmember_number).astype(int)
    RMSE_abundance = np.zeros(endmember_number)
    SAD_endmember = np.zeros(endmember_number)

    for i in range(0, endmember_number):
        for j in range(0, endmember_number):
            RMSE_matrix[i, j] = AbundanceRmse(
                abundance_input[i, :, :], abundance_GT_input[j, :, :]
            )
            SAD_matrix[i, j] = SAD_distance(endmember_input[:, i], endmember_GT[:, j])

        RMSE_index[i] = np.argmin(RMSE_matrix[i, :])
        SAD_index[i] = np.argmin(SAD_matrix[i, :])
        RMSE_abundance[i] = np.min(RMSE_matrix[i, :])
        SAD_endmember[i] = np.min(SAD_matrix[i, :])

    abundance_input[np.arange(endmember_number), :, :] = abundance_input[
        RMSE_index, :, :
    ]
    endmember_input[:, np.arange(endmember_number)] = endmember_input[:, SAD_index]

    return abundance_input, endmember_input, RMSE_abundance, SAD_endmember


class load_data(torch.utils.data.Dataset):
    def __init__(self, img, gt, transform=None):
        self.img = img.float()
        self.gt = gt.float()
        self.transform = transform

    def __getitem__(self, idx):
        return self.img, self.gt

    def __len__(self):
        return 1


# calculate RMSE of abundance
def AbundanceRmse(inputsrc, inputref):
    rmse = np.sqrt(((inputsrc - inputref) ** 2).mean())
    return rmse


# calculate SAD of endmember
def SAD_distance(src, ref):
    cos_sim = np.dot(src, ref) / (np.linalg.norm(src) * np.linalg.norm(ref))
    SAD_sim = np.arccos(cos_sim)
    return SAD_sim


# SAD loss of reconstruction
def reconstruction_SADloss(output, target):

    _, band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0))
    abundance_loss = torch.mean(abundance_loss)

    return abundance_loss


MSE = torch.nn.MSELoss(size_average=True)

# load data
train_dataset = load_data(
    img=original_HSI, gt=abundance_GT, transform=transforms.ToTensor()
)
# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=False
)

net = multiStageUnmixing(L=band_Number, R=endmember_number, img_size=col).cuda()
# weight init
net.apply(net.weights_init)

# decoder weight init by VCA
model_dict = net.state_dict()
model_dict["decoderlayer4.0.weight"] = endmember_init

net.load_state_dict(model_dict)


# optimizer upgrade: You can set special lr and weight_decay for the linear part of decoder
def set_optimizer(model, lr_base, decay):
    slow_params = map(id, model.decoderlayer4.parameters())
    else_params = filter(lambda addr: id(addr) not in slow_params, model.parameters())
    optimizer = torch.optim.Adam([
        {'params': model.decoderlayer4.parameters(), 'lr': 1e-5},
        {'params': else_params}], lr=lr_base, weight_decay=decay
    )
    return optimizer


# optimizer
optimizer = set_optimizer(model=net, lr_base=learning_rate, decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.6)
apply_clamp_inst1 = NonZeroClipper()

train_losses = []
abundance_losses = []
mse_losses = []

'''Train the model'''
for epoch in range(EPOCH):
    for i, (x, y) in enumerate(train_loader):
        scheduler.step()
        x = x.cuda()
        net.train().cuda()

        en_abundance, reconstruction_result, b_result = net(x)

        abundanceLoss = reconstruction_SADloss(x, reconstruction_result)

        MSELoss = MSE(x, reconstruction_result)

        ALoss = abundanceLoss
        BLoss = MSELoss

        total_loss = ALoss + (alpha * BLoss)
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=10, norm_type=1)
        optimizer.zero_grad()
        net.decoderlayer4.apply(apply_clamp_inst1)
        total_loss.backward()
        optimizer.step()

        train_losses.append(total_loss.item())
        abundance_losses.append(ALoss.item())
        mse_losses.append(BLoss.item())

        if epoch % 100 == 0:
            print(
                "Epoch:",
                epoch,
                "| Abundanceloss: %.4f" % ALoss.cpu().data.numpy(),
                "| MSEloss: %.4f" % (alpha * BLoss).cpu().data.numpy(),
                "| total_loss: %.4f" % total_loss.cpu().data.numpy(),
            )

# 训练结束后绘制损失曲线
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Total Loss', color='blue', linewidth=2)
plt.plot(abundance_losses, label='Abundance Loss', color='red', linestyle='--', linewidth=2)
plt.plot(mse_losses, label='MSE Loss', color='green', linestyle='-.', linewidth=2)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

net.eval()


en_abundance, reconstruction_result, b_result = net(x)

decoder_para = net.state_dict()["decoderlayer4.0.weight"].cpu().numpy()
decoder_para = np.mean(np.mean(decoder_para, -1), -1)

en_abundance, abundance_GT = norm_abundance_GT(en_abundance, abundance_GT)
decoder_para, GT_endmember = norm_endmember(decoder_para, GT_endmember)

en_abundance, decoder_para, RMSE_abundance, SAD_endmember = arange_A_E(
    en_abundance, abundance_GT, decoder_para, GT_endmember
)
print("RMSE", RMSE_abundance)
print("mean_RMSE", RMSE_abundance.mean())
print("endmember_SAD", SAD_endmember)
print("mean_SAD", SAD_endmember.mean())

# Save the results to the current directory.
sio.savemat('Name_of_Your_Result.mat', {'M_pre': decoder_para, 'A_pre': en_abundance})

# Plot the results
plot_abundance(en_abundance, abundance_GT)
plot_endmember(decoder_para, GT_endmember)


'''Evaluate the prediction of nonlinear coefficients b (on the PPNMM-generated synthetic datasets)'''
b_predict = torch.flatten(b_result, 1)
b_true = b_true.cuda()
b_true = b_true.view(1, col, col)
b_true = torch.flatten(b_true, 1)
MSE_b = torch.sum((b_predict - b_true) ** 2, 1)/10000
MSE_b = MSE_b.item()
RMSE_b = pow(MSE_b, 0.5)
print("RMSE_b", RMSE_b)
print(b_result)
b_predict = b_predict.cpu().detach().numpy()
sio.savemat('b_result_1.mat', {'b_result': b_predict})
