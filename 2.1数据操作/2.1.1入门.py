import numpy as np
import torch

'''
2022年10月22日14:46:42
'''
#
x = torch.arange(12)  # 生成0-12的元素，是一个向量
x
print(x)
# x.shape 来访问形状
print(x.shape)
# 元素的种类,是一个标量
print(x.numel())
# 改变数组的形状，不改变元素的数量和元素值，调用reshape函数
X = x.reshape(3, 4)  # reshape为三行四列
x.reshape(-1, 4)  # 可以用-1自动计算出维度，得到列的值自动计算出行
# 创建全为0和全为1的张量，和随机张量
torch.zeros((2, 3, 4))
torch.ones((2, 3, 4))
torch.randn(3, 4)  # 每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样。
print(X)
# 创建一个二维数组，
X = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])  # list of list 列表嵌套列表
print(X)
print(X.shape)
# 三维数组再加括号
X = torch.tensor([[[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]])  # list of list 列表嵌套列表
print(X)
print(X.shape)

# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 2.1.2运算:所有的运算都是按照元素进行的
print("第二节")
x = torch.tensor([1.0, 2, 4, 8])  # 特意创建浮点元素数组
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算
print(x + y, x - y, x * y, x / y, x ** y)
# 也可以把多个张量连结（concatenate）在一起， 把它们端对端地叠起来形成一个更大的张量。
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))  # 类型是浮点型
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(X, Y)
# cat 合并张量
torch.cat((X, Y), dim=0)  # 按行合并 轴-0，形状的第一个元素
torch.cat((X, Y), dim=1)  # 按列合并 轴-1，形状的第二个元素 ，即四列
print(torch.cat((X, Y), dim=1))
# 通过逻辑运算符构建二元张量
print(X == Y)
# 张量元素的求和
print(X.sum())
# 继承numpy
a = torch.arange(3).reshape((3, 1))  # 三行一列
b = torch.arange(2).reshape((1, 2))  # 一行两列
print(a, b)
print(a + b)  # 两个张量相加（形状不一样）将a复制为3×2的矩阵，b复制为3*2的矩阵再相加
# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 索引和切片
print(X)
print(X[-1], X[1:3])  # 用[-1]选择最后一个元素，可以用[1:3]选择第二个和第三个元素：
# 张量元素赋值 除读取外，我们还可以通过指定索引来将元素写入矩阵。
X[1, 2] = 9
X[0:2, :] = 12  # 0-1行，所有的列，赋值为12
# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 节省内存,防止特别大的矩阵不断赋值导致内存溢出
before = id(Y)  # id类似于c++的指针
Y = Y + X  # 进行操作之后改变id
print(id(Y) == before)  # 输出false 新的y的id不等于以前
# 执行原地操作
Z = torch.zeros_like(Y)  # 与y的shape和类型是一样的，但是所有的元素是0
print('id(Z):', id(Z))
Z[:] = X + Y  # Z里面所有的元素=x+y
print('id(Z):', id(Z))
# 后续计算中没有重复使用X， 我们也可以使用X[:] = X + Y或X += Y来减少操作的内存开销。
before = id(X)
X += Y  # 直接把Y的值加进X
print(id(X) == before)
# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 转换为其他python对象
A = X.numpy()
B = torch.tensor(A)
print(type(A))
print(type(B))
# 大小为1的张量转换为Python标量，我们可以调用item函数或Python的内置函数。只能大小为一变
a = np.array([3.5])
print(a)
print(a.item())  # numpy的一个浮点数
print(float(a), int(a))  # 浮点数，整数

