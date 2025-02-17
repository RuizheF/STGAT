import torch
import random
import os
from torch_geometric.data import DataLoader
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# from Agent_SL import GCN, SGFormer, TCN_Net
from STGAT_tsd import GAT_TCN
from gen_data_tsd import gen_GNN_data
plt.rcParams['font.sans-serif'] = ['SimHei']	# 显示中文
plt.rcParams['axes.unicode_minus'] = False		# 显示负号
current_path = os.path.dirname(os.path.abspath(__file__)) # 获取当前脚本所在的项目根目录
root_path = os.path.dirname(current_path)
print("项目根目录路径：", root_path)
import numpy as np
import pandas as pd
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

fix_seed(50)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Gdata_list, split, max_value, min_value = gen_GNN_data()
data_set = Gdata_list
dataset_size = len(data_set)
train_ratio = 0.15  # 训练集占的比例
val_test_ratio = 0.37  # 验证集和测试集在剩余部分中的比例，验证集占一半
train_size = int(dataset_size * train_ratio)
val_size = int(train_size * val_test_ratio)
train_dataset = data_set[:-val_size]
val_dataset = data_set[-val_size:]
test_dataset = val_dataset
print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))
train_batch = 16
val_batch = 16
# 重新定义 DataLoader，加载整个数据集进行预测
full_train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=False)
full_val_loader = DataLoader(val_dataset, batch_size=val_batch, shuffle=False)
full_test_loader = DataLoader(test_dataset, batch_size=val_batch, shuffle=False)

gat = GAT_TCN(462, 1, 128, 1, 3, 2, 3)
model = gat.to(device)
# 加载训练好的模型权重
best_model = gat.to(device)
best_model.load_state_dict(torch.load('best_model_gat_tcn_US_fixed2.pt'))

def predict_and_calculate(model, data_loader):
    model.eval()
    all_predictions = []
    all_gourujia = []
    all_std_dev_next = []
    all_true = []
    all_zhangting = []
    all_zhishu = []
    all_Nasdaq_zhishu = []
    all_DJIA_zhishu = []
    all_riqi = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data)
            y = data.shouchujia.reshape(-1, 300)
            all_true.extend(y.cpu().numpy().tolist())
            all_predictions.extend(out.cpu().numpy().tolist())
            all_gourujia.extend(data.gourujia.reshape(-1, 300).cpu().numpy().tolist())
            all_std_dev_next.extend(data.std_dev_next.reshape(-1, 300).cpu().numpy().tolist())
            all_zhangting.extend(data.zhangting.reshape(-1, 300).cpu().numpy().tolist())
            all_zhishu.extend(data.biaopuzhishu)
            all_Nasdaq_zhishu.extend(data.Nasdaq_zhishu)
            all_DJIA_zhishu.extend(data.DJIA_zhishu)
            all_riqi.extend(data.riqi)


    true_values = np.array(all_true)
    predictions = np.array(all_predictions)
    gourujia = np.array(all_gourujia)
    std_dev_next = np.array(all_std_dev_next)
    zhangting = np.array(all_zhangting)
    biaopuzhishu = np.array(all_zhishu)
    Nasdaq_zhishu = np.array(all_Nasdaq_zhishu)
    DJIA_zhishu = np.array(all_DJIA_zhishu)
    riqi = np.array(all_riqi)

    std_dev_next[std_dev_next == 0] = 1e-5

    earnings_rate = (predictions - gourujia) / gourujia * 100
    true_earnings_rate = (true_values - gourujia) / gourujia * 100

    # 创建掩码，过滤收益率和涨停值
    # mask_earnings_rate = earnings_rate <= 9.9
    mask_zhangting = zhangting <= 9.9

    # 结合掩码
    combined_mask = mask_zhangting

    filtered_earnings_rate = np.where(combined_mask, earnings_rate, np.nan)
    filtered_true_earnings_rate = np.where(combined_mask, true_earnings_rate, np.nan)
    filtered_std_dev_next = np.where(combined_mask, std_dev_next, np.nan)

    portfolio_value = np.nan_to_num(filtered_earnings_rate / filtered_std_dev_next, nan=0.0)
    true_portfolio_value = np.nan_to_num(filtered_true_earnings_rate / filtered_std_dev_next, nan=0.0)

    return filtered_earnings_rate, portfolio_value, gourujia, predictions, filtered_true_earnings_rate, true_portfolio_value, true_values, biaopuzhishu, Nasdaq_zhishu, DJIA_zhishu, riqi

def select_top_stocks_and_calculate_shares(portfolio_value, gourujia, predictions, true_portfolio_value,true_earnings_rate, output_folder):
    top_stocks_per_day = []
    shares_per_day = []

    for day in range(portfolio_value.shape[0]):
        # 选择组合价值最高的10只股票
        top_indices = np.argsort(portfolio_value[day])[-10:][::-1]
        top_values = portfolio_value[day][top_indices]

        # 计算投资份额比例
        total_value = np.sum(top_values)
        shares = top_values / total_value

        # 检查份额比例和是否为1，如果不是则进行归一化
        if not np.isclose(np.sum(shares), 1.0):
            shares = shares / np.sum(shares)

        top_stocks_per_day.append(top_indices)
        shares_per_day.append(shares)

    # 将选定的股票的 gourujia, predictions 和 true_portfolio_value 结果保存到CSV文件
    for day, (top_indices, shares) in enumerate(zip(top_stocks_per_day, shares_per_day)):
        selected_gourujia = gourujia[day][top_indices]
        selected_predictions = predictions[day][top_indices]
        selected_true_portfolio_value = true_portfolio_value[day][top_indices]
        selected_true_earnings_rate = true_earnings_rate[day][top_indices]

        # 将结果保存到CSV文件
        df = pd.DataFrame({
            'Stock_Index': top_indices,
            'Gourujia': selected_gourujia.flatten(),
            'Predictions': selected_predictions.flatten(),
            'Shares': shares,
            'True_Portfolio_Value': selected_true_portfolio_value.flatten(),
            'True_Earnings_Rate': selected_true_earnings_rate.flatten(),


        })
        df.to_csv(f'{output_folder}/day_{day + 1}_top_stocks_and_shares.csv', index=False, float_format='%.6f')

    return top_stocks_per_day, shares_per_day

# 获取训练数据集的组合价值、估值和股价
train_earnings_rate, train_portfolio_value, train_gourujia, train_predictions,true_train_earnings_rate, true_train_portfolio_value, true_train_values,biaopuzhishu_train,Nasdaq_zhishu_train,DJIA_zhishu_train,riqi_train = predict_and_calculate(best_model, full_train_loader)
# 获取验证数据集的组合价值、估值和股价
val_earnings_rate, val_portfolio_value, val_gourujia, val_predictions, true_val_earnings_rate, true_val_portfolio_value, true_val_values,biaopuzhishu_val,Nasdaq_zhishu_val,DJIA_zhishu_val,riqi_val  = predict_and_calculate(best_model, full_val_loader)
test_earnings_rate, test_portfolio_value, test_gourujia, test_predictions, true_test_earnings_rate, true_test_portfolio_value, true_test_values ,biaopuzhishu_test, Nasdaq_zhishu_test,DJIA_zhishu_test,riqi_test= predict_and_calculate(best_model, full_test_loader)

# 选择每个交易日组合价值最高的10只股票并计算份额比例，并保存相关信息到CSV文件
train_top_stocks, train_shares = select_top_stocks_and_calculate_shares(train_portfolio_value, train_gourujia, train_predictions,true_train_portfolio_value,true_train_earnings_rate, '../output/train')
val_top_stocks, val_shares = select_top_stocks_and_calculate_shares(val_portfolio_value, val_gourujia, val_predictions,true_val_portfolio_value, true_val_earnings_rate, '../output/val')
test_top_stocks, test_shares = select_top_stocks_and_calculate_shares(test_portfolio_value, test_gourujia, test_predictions,true_test_portfolio_value, true_test_earnings_rate, '../output/test')
print("每个交易日组合价值最高的10只股票及其份额比例已保存到CSV文件。")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def read_stock_data_for_days(folder_path):
    """
    读取文件夹中每个交易日的股票数据文件并返回一个字典，键是日期，值是对应的DataFrame对象
    """
    stock_data_dict = {}
    for file_name in os.listdir(folder_path):
        if file_name.startswith('day_') and file_name.endswith('_top_stocks_and_shares.csv'):
            day = int(file_name.split('_')[1])
            file_path = os.path.join(folder_path, file_name)
            stock_data_dict[day] = pd.read_csv(file_path)
    return stock_data_dict

def simulate_trading(stock_data, initial_capital):
    """
    模拟一日股票交易过程
    """
    remain_capital = initial_capital
    remain_capital_true = initial_capital
    capital = initial_capital
    shares_dict = {}
    sold_dict = {}
    bought_dict = {}
    sold_dict_true = {}
    for index, row in stock_data.iterrows():
        stock_index = row['Stock_Index']
        gourujia = row['Gourujia']
        predictions = row['Predictions']
        shares = row['Shares']
        True_Value = row['True_Portfolio_Value']
        Earnings_Rate = row['True_Earnings_Rate']

        # 计算购买股票数
        stock_amount = int(shares * capital / gourujia)
        shares_dict[stock_index] = stock_amount
        # 计算花费
        total_cost = gourujia * stock_amount
        # 更新剩余资金
        remain_capital -= total_cost
        remain_capital_true-=total_cost
        # 记录本次买入情况
        bought_dict[stock_index] = total_cost
    for index, row in stock_data.iterrows():
        stock_index = row['Stock_Index']
        gourujia = row['Gourujia']
        predictions = row['Predictions']
        shares = row['Shares']
        true_value = row['True_Portfolio_Value']
        earnings_rate = row['True_Earnings_Rate']

        # 计算卖出金额
        sell_amount = shares_dict[stock_index] * predictions
        sell_amount_true = shares_dict[stock_index] * (gourujia * (1 + earnings_rate / 100))
        # 更新剩余资金
        remain_capital += sell_amount
        remain_capital_true += sell_amount_true

        # 记录本次卖出情况
        sold_dict[stock_index] = sell_amount
        sold_dict_true[stock_index] = sell_amount_true

    # 计算每日收益率
    daily_earnings_rate = (remain_capital - capital) / capital * 100
    daily_earnings_rate_true = (remain_capital_true - initial_capital) / initial_capital * 100

    return remain_capital, daily_earnings_rate,remain_capital_true,daily_earnings_rate_true


def simulate_trading_for_days(stock_data_dict, initial_capital=100000):
    """
    模拟多日股票交易过程
    """
    capital_dict = {}
    earnings_rate_dict = {}
    capital_dict_true = {}
    earnings_rate_dict_true = {}
    capital = initial_capital

    # 按阿拉伯数字顺序对天数进行排序
    sorted_days = sorted(stock_data_dict.keys(), key=lambda x: int(x))

    for day in sorted_days:
        stock_data = stock_data_dict[day]
        capital, earnings_rate, capital_true, earnings_rate_true = simulate_trading(stock_data, initial_capital)
        capital_dict[day] = capital
        earnings_rate_dict[day] = earnings_rate
        capital_dict_true[day] = capital_true
        earnings_rate_dict_true[day] = earnings_rate_true

    return capital_dict, earnings_rate_dict, capital_dict_true, earnings_rate_dict_true


# 读取文件夹中每个交易日的股票数据
train_stock_data_dict = read_stock_data_for_days('../output/train')
val_stock_data_dict = read_stock_data_for_days('../output/val')
test_stock_data_dict = read_stock_data_for_days('../output/test')

# 模拟交易过程
test_capital_dict, test_earnings_rate_dict,test_capital_dict_true, test_earnings_rate_dict_true = simulate_trading_for_days(test_stock_data_dict)
test_capital_moving_avg = [value for value in test_capital_dict.values()]
test_capital_moving_avg_true = [value for value in test_capital_dict_true.values()]

test_earnings_rate = [value for value in test_earnings_rate_dict_true.values()]

# 提取并过滤测试集和验证集的收益率
# train_earnings_rate = [(1+value/100) for value in train_earnings_rate_dict_true.values() if value <= 40]
# val_earnings_rate = [(1+value/100) for value in val_earnings_rate_dict_true.values() if value <= 40]
test_earnings_rate = [(1+value/100) for value in test_earnings_rate_dict_true.values() ]

test_biaopuzhishu = [(1+value/100) for value in biaopuzhishu_test]
test_Nasdaq_zhishu = [(1+value/100) for value in Nasdaq_zhishu_test]
test_DJIA_zhishu = [(1+value/100) for value in DJIA_zhishu_test]

# 累积收益率计算
# train_cumulative_earnings_rate = np.cumprod(train_earnings_rate)
# val_cumulative_earnings_rate = np.cumprod(val_earnings_rate)
test_cumulative_earnings_rate = np.cumprod(test_earnings_rate)*100
test_cumulative_biaopuzhishu = np.cumprod(test_biaopuzhishu)*100
test_cumulative_Nasdaq_zhishu = np.cumprod(test_Nasdaq_zhishu)*100
test_cumulative_DJIA_zhishu = np.cumprod(test_DJIA_zhishu)*100

# 绘制训练集累积收益率曲线图
plt.figure(figsize=(20, 10))
plt.plot(test_cumulative_earnings_rate, label='Ours', linestyle='-', marker='s', color='blue')
plt.plot(test_cumulative_biaopuzhishu, label='SPX500', linestyle='--', marker='o', color='red')
plt.plot(test_cumulative_Nasdaq_zhishu,label='Nasdaq',color='green', linestyle='-.', marker='*')
plt.plot(test_cumulative_DJIA_zhishu,label='DJIA',color='black', linestyle=':', marker='^')
plt.xticks(ticks=range(len(riqi_test)), labels=riqi_test, rotation=45)  # rotation=45 将日期标签旋转45度，便于阅读

plt.xlabel('Trading Day')
plt.ylabel('Cumulative Earnings Rate(100%)')
plt.title('Test Cumulative Earnings Rate Change Over Days')
plt.legend()
plt.grid(True)
plt.show()


# 确保所有数据都是一维数组或列表
riqi_test = list(riqi_test.reshape(25))
earnings_rate_values = list(test_earnings_rate_dict_true.values())
biaopuzhishu_test = list(biaopuzhishu_test.reshape(25))
Nasdaq_zhishu_test = list(Nasdaq_zhishu_test.reshape(25))
DJIA_zhishu_test = list(DJIA_zhishu_test.reshape(25))

# 将每日收益率保存到CSV文件
test_daily_rates_df = pd.DataFrame({
    'Day': riqi_test,
    'Earnings Rate': earnings_rate_values,
    'SPX500 Rate': biaopuzhishu_test,  # 假设 biaopuzhishu_test 是每日百分比变化
    'Nasdaq Rate': Nasdaq_zhishu_test,
    'DJIA Rate': DJIA_zhishu_test
})
test_daily_rates_df.to_csv('../output/test_earnings_rate_fixed2.csv', index=False, encoding='gb18030')


# 将累积收益率保存到CSV文件
test_cumulative_rates_df = pd.DataFrame({
    'Day': riqi_test,
    'Cumulative Earnings Rate': test_cumulative_earnings_rate,
    'Cumulative SPX500 Rate': test_cumulative_biaopuzhishu,
    'Cumulative Nasdaq Rate': test_cumulative_Nasdaq_zhishu,
    'Cumulative DJIA Rate': test_cumulative_DJIA_zhishu
})
test_cumulative_rates_df.to_csv('../output/test_cumulative_rates_fixed2.csv', index=False, encoding='gb18030')

print("CSV files have been saved.")
