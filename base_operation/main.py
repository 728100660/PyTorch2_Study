import torch
import numpy as np


def func1():
    v = torch.tensor([1, 2])
    print(v)
    v2 = torch.tensor([[1,2], [3,4]])
    print(v2)
    v3 = torch.tensor(np.array([1, 2]))
    print(v3)


def func2():
    # 全1的矩阵
    print(torch.ones(2, 3))
    # 全0的矩阵
    print(torch.zeros(2, 3))
    # 填充为指定值
    print(torch.full((2, 3), 11))
    # 对角矩阵，填充值默认为1
    print(torch.eye(3, 2))


def func3():
    # 创建但不初始化
    print(torch.empty(2, 3))
    # 标准正态分布（均值为0，方差为1）
    print(torch.randn(2, 3))
    # 0-1的均匀分布(均匀是说每个数字出现的可能性都是均匀的，就是随机)
    print(torch.rand(2, 3))

    print(torch.randint(0, 10, (2, 3)))


def func4():
    print(torch.arange(0, 10, 2))
    print(torch.linspace(0, 10, 4))


def func5():
    print(torch.FloatTensor(2, 3))

    print(torch.randn(2, 3).to(torch.float64).dtype)


def main():
    # func1()
    # func2()
    # func3()
    # func4()
    func5()


if __name__ == '__main__':
    main()