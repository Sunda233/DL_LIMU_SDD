import os  # 文件的操作
import pandas as pd  # 可以读取csv文件
import torch

'''
2022年10月23日23:52:57
简单的数据预处理
'''

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 创建一个人工的数据集并储存在CSV文件  CSV:每一行是一个数据，每一个域是由逗号分开的
os.makedirs(os.path.join('..', 'data'), exist_ok=True)  # 创建一个文件夹
data_file = os.path.join('..', 'data', 'house_tiny.csv')  # 文件名字
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名  （房间，路线，价格）
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# panda库，读取文件
data = pd.read_csv(data_file)
print(data)
# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 处理缺失的数据，NaN”项代表缺失值。 为了处理缺失的数据，典型的方法包括插值法和删除法
# 插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。 在这里，我们将考虑插值法。
# 通过位置索引iloc，我们将data分成inputs和outputs， 其中前者为data的前两列，而后者为data的最后一列。 对于inputs中缺少的数值，我们用同一列的均值替换“NaN”项
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]  # 假设把数据分为输入特征和输出特征
print(inputs)
print("-------")
print(outputs)
#
inputs = inputs.fillna(inputs.mean())  # fillna:所有的na的域填一个值（mean），填成剩下不是nan的值得均值
print(inputs)
# 对于inputs中的类别值或离散值，我们将“NaN”视为一个类别。
inputs = pd.get_dummies(inputs, dummy_na=True)  # 由此函数转换为类别，dummy_na=True：na也要加一个特别的类，所以是两类
print("转换为类别：")
print(inputs)
# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 现在inputs和outputs中的所有条目都是数值类型，它们可以转换为张量格式。
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)  # torch.float64,传统python会用,对于深度学习计算一般比较慢，用32位浮点数
print(X, y)

