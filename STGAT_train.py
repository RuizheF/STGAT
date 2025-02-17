import numpy as np
import torch
import torch.nn as nn
import random
import os
from torch_geometric.data import DataLoader
import pandas as pd
from matplotlib import pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")
# from Agent_SL import GCN,SGFormer,TCN_Net
from torch.utils.data import random_split
from STGAT_tsd import GAT_TCN
from gen_data_tsd import gen_GNN_data
# from gen_data import gen_GNN_data
# from GAT import GAT_TCN

current_path = os.path.dirname(os.path.abspath(__file__)) # 获取当前脚本所在的项目根目录
root_path = os.path.dirname(current_path)
from sklearn.model_selection import KFold

print("项目根目录路径：", root_path)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# edgeindex = [[0, 1], [0, 2], [1, 3], [2, 3], [1, 4], [1, 5], [3, 5], [4, 6], [5, 6], [5, 27], [5, 7],
# [7, 27], [27, 26], [26, 29], [29, 28], [26, 28], [26, 24], [24, 25], [5, 8], [8, 10], [8, 9], [5, 9], [9, 20],
# [20, 21], [9, 16], [15, 16], [3, 11], [11, 12], [11, 17], [11, 15], [17, 18], [18, 19], [9, 19], [9, 23], [11, 13],
# [13, 14], [14, 22], [22, 23], [23, 24], [1, 0], [2, 0], [3, 1], [3, 2], [4, 1], [5, 1], [5, 3], [6, 4], [6, 5], [27, 5],
# [7, 5], [27, 7], [26, 27], [29, 26],  [28, 29], [28, 26], [24, 26], [25, 24], [8, 5], [10, 8], [9, 8], [9, 5], [20, 9],
# [21, 20], [16, 9], [16, 15], [11, 3], [12, 11], [17, 11], [15, 11], [18, 17], [19, 18], [19, 9], [23, 9], [13, 11], [14, 13], [22, 14], [23, 22], [24, 23]]
# edge_index = torch.from_numpy(np.array(edgeindex).transpose()).long()

fix_seed(50)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
Gdata_list, split, max_value, min_value = gen_GNN_data()

data_set = Gdata_list
dataset_size = len(data_set)

# 收益率划分数据集方法
train_ratio = 0.15  # 训练集占的比例
val_test_ratio = 0.38  # 验证集和测试集在剩余部分中的比例，验证集占一半
train_size = int(dataset_size * train_ratio)
val_size = int(train_size * val_test_ratio)
train_dataset = data_set[:-val_size]
val_dataset = data_set[-val_size:]
test_dataset = val_dataset
train_dataset_all = train_dataset

# train_ratio = 0.9  # 训练集占的比例
# val_test_ratio = 0.5  # 验证集和测试集在剩余部分中的比例，验证集占一半
#
# # 计算各部分的大小
# train_size_all = int(dataset_size * train_ratio)
#
# # 固定划分
#
# val_size = int(train_size_all * 0.5)
# train_size = train_size_all - val_size
# test_size = int(dataset_size - train_size_all)
# train_dataset_all = data_set[:train_size_all]
# train_dataset, val_dataset = random_split(train_dataset_all, [train_size, val_size])
# # val_dataset = data_set[180:210]
# test_dataset = data_set[train_size_all:]

# # 随机划分
# train_size = int(train_size_all * 0.5)
# val_size = int(train_size_all * 0.5)
# test_size = int(dataset_size - train_size - val_size)
# train_dataset, val_dataset,test_dataset = random_split(data_set, [train_size, val_size, test_size])
# train_dataset_all = train_dataset + val_dataset

# 计算各部分的大小
# train_size = int(dataset_size * train_ratio)
# val_size = int(train_size * val_test_ratio)
# # val_test_size = dataset_size - train_size
# # val_size = int(val_test_size * val_test_ratio)
# # test_size = val_test_size - val_size
#
# # 顺序划分数据集
# train_dataset = data_set[:-25]
# val_dataset = data_set[-25:]
#test_dataset = data_set[train_size + val_size:]
# # 顺序划分数据集
# train_dataset = data_set[:train_size]
# val_dataset = data_set[train_size:train_size + val_size]
# test_dataset = data_set[train_size + val_size:]
print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))
#
# # 获取测试集中每个数据在原始数据集中的索引
# test_indices = test_dataset.indices
#
# # 打印测试集的索引
# print("Test dataset indices:", test_indices)
#
# # 验证索引对应的值
# for idx in test_indices:
#     print(f"Index: {idx}, gouru_Value: {Gdata_list[idx].goururi}")
#     print(f"Index: {idx}, kaishi_Value: {Gdata_list[idx].kaishiri}")
# # print("Test size:", len(test_dataset))
# data = []
# for idx in test_indices:
#     data.append([idx, Gdata_list[idx].goururi, Gdata_list[idx].kaishiri])
#
# # 使用 pandas 创建 DataFrame 并写入 CSV 文件
# df = pd.DataFrame(data, columns=['Index', 'Gouru_Value', 'Kaishi_Value'])
# output_file = '../output/riqi.csv'
# df.to_csv(output_file, index=False)
#
# print(f"Data written to {output_file}")
train_batch = 16
val_batch = 16

K = 5
kf = KFold(n_splits=K, shuffle=True, random_state=2)
# in_channels = 288
# Gout_put_channels = 128
# Ghidden_channels = 128
#
# hidden_channels = Gout_put_channels
# out_channels = 1728
#
# in_channels = split
# out_channels = 6*split

aggregate = 'cat'
lr = 1e-3
ours_weight_decay = 5e-3
weight_decay = 5e-3
epochs = 100
val_min_num = 50

in_size = 30
out_channels = 304


# TCN = TCN_Net(in_size,hidden_channels,split)
# gcn = GCN(in_channels,Ghidden_channels,Gout_put_channels).to(device)
# model = SGFormer(in_channels, hidden_channels, out_channels, aggregate, gcn, TCN).to(device)
gat = GAT_TCN(304, 1, 128, 1, 3, 2, 3)

# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
criterion_2 = nn.L1Loss()
criterion = torch.nn.MSELoss()
# criterion = criterion_2
# optimizer = torch.optim.Adam([
#     {'params': model.params1},
#     {'params': model.params2}
# ],
#     lr=lr)
# optimizer = optim.NAdam([
#     {'params': model.params1},
#     {'params': model.params2},
#     {'params': model.params3},
#     {'params': model.params4}], lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=7, verbose=True)

# 梯度累计参数
accumulation_steps = 4
effective_batch_size = train_batch * accumulation_steps
# optimizer = torch.optim.SophiaG(model.parameters(), lr=0.001)
# optimizer = torch.optim.SGD([
#     {'params': model.params1},
#     {'params': model.params2}
# ],
#     lr=lr,momentum=0.5)
def train():
    model.train()
    total_loss = 0
    for step, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        # out = model(data) * (data.max.unsqueeze(1) - data.min.unsqueeze(1)) + data.min.unsqueeze(1)
        # y = (data.shouchujia.reshape(-1, 304))* (data.max.unsqueeze(1) - data.min.unsqueeze(1)) + data.min.unsqueeze(1)
        # out = model(data)*data.guiyi_std.unsqueeze(1)+data.mean_value.unsqueeze(1)
        # y = data.shouchujia.reshape(-1, 304)*data.guiyi_std.unsqueeze(1)+data.mean_value.unsqueeze(1)
        out = model(data)
        y = data.shouchujia.reshape(-1, 300)
        loss = criterion(out, y)
        loss.backward()
        # if (step+1)% accumulation_steps == 0:

        optimizer.step()
        L1loss = criterion_2(out, y)
        # total_loss += loss.item()
        total_loss += L1loss.item()
    return total_loss / len(train_loader)
def validate(model_xc):
    model_xc.eval()
    total_loss = 0
    all_predictions = []  # 用于保存验证过程中的所有预测值
    all_targets = []      # 用于保存验证过程中的所有真实值

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            # out = model(data) * (data.max.unsqueeze(1) - data.min.unsqueeze(1)) + data.min.unsqueeze(1)
            # y = (data.shouchujia.reshape(-1, 304)) * (
            #       data.max.unsqueeze(1) - data.min.unsqueeze(1)) + data.min.unsqueeze(1)
            # out = model(data) * data.guiyi_std.unsqueeze(1) + data.mean_value.unsqueeze(1)
            # y = data.shouchujia.reshape(-1, 304) * data.guiyi_std.unsqueeze(1) + data.mean_value.unsqueeze(1)
            out = model(data)
            y = data.shouchujia.reshape(-1, 300)
            loss = criterion(out, y)
            L1loss = criterion_2(out, y)
            # total_loss += loss.item()
            total_loss += L1loss.item()
            all_predictions.extend(out.cpu().numpy().tolist()) # 保存预测值
            all_targets.extend(y.cpu().numpy().tolist())       # 保存真实值

    return total_loss / len(val_loader), all_predictions, all_targets


val_predictions = []
val_targets = []
val_loss_list = []
train_loss_list = []

# 记录最佳验证损失
best_val_loss = float('inf')
best_epoch = 0
best_model_state_dict = None

for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset_all)):
    print(f'Fold {fold + 1}/{K}')

    train_subset = [train_dataset_all[i] for i in train_idx]
    val_subset = [train_dataset_all[i] for i in val_idx]

    train_loader = DataLoader(train_subset, batch_size=train_batch, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=val_batch, shuffle=False)

    model = gat.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=7, verbose=True)

    for epoch in range(epochs):  # number of epochs
        train_loss = train()
        val_loss, epoch_val_preds, epoch_val_targets = validate(model)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        if epoch > val_min_num:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state_dict = model.state_dict()
                val_predictions = epoch_val_preds
                val_targets = epoch_val_targets
                #print(f'New best model found at epoch {epoch} with validation loss {val_loss:.4f}.')
            scheduler.step(val_loss)

        print(f'Epoch: {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

best_model_state_dict = model.state_dict()
if best_model_state_dict is not None:
    torch.save(best_model_state_dict, 'best_model_TGT_gat_tcn_A_fixed2.pt')

print(f'Best Test Loss: {best_val_loss:.4f}')
# print(f'Best Test Loss: {best_val_loss:.4f}')
# print(f'Best Test Loss at epoch: {best_epoch}')
# print(f'Best Test Predictions: {val_predictions}')
# print(f'Best Test Targets: {val_targets}')
# 将测试预测值和真实值保存到DataFrame
val_df = pd.DataFrame({
    'Val_Predictions': [item for sublist in val_predictions for item in sublist],
    'Val_Targets': [item for sublist in val_targets for item in sublist]
})
# print(val_df)
val_df['Val_Predictions'] = val_df['Val_Predictions'].astype(float)
val_df['Val_Targets'] = val_df['Val_Targets'].astype(float)
# 保存到CSV文件
val_df.to_csv('prediction_gat_tcn_A_fixed2.csv', index=False,float_format='%.6f')
print("验证预测值和真实值已保存到CSV文件。")

train_loss_list = train_loss_list[val_min_num:]
val_loss_list = val_loss_list[val_min_num:]

episodes_train_list = list(range(len(train_loss_list)))
episodes_val_list = list(range(len(val_loss_list)))
# plt.plot(episodes_list, return_list, label='Returns', color='blue')
plt.plot(episodes_train_list, train_loss_list, label='train_loss_change', color='green')
plt.plot(episodes_val_list, val_loss_list, label='val_loss_change', color='red')
# plt.plot(episodes_list, return_pdemand_cost_list, label='Pdemand Cost Returns', color='red')
plt.xlabel('Episodes')
plt.ylabel('loss')
plt.legend()
plt.show()
