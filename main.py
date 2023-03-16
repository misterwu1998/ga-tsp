import numpy as np
import config as conf
from ga import Ga, solveTSP
import matplotlib.pyplot as plt

config = conf.get_config()


def build_dist_mat(input_list):
    n = config.city_num
    dist_mat = np.zeros([n, n])
    for i in range(n):
        for j in range(i + 1, n):
            d = input_list[i, :] - input_list[j, :]
            # 计算点积
            dist_mat[i, j] = np.dot(d, d)
            dist_mat[j, i] = dist_mat[i, j]
    return dist_mat


# 城市坐标
city_pos_list = np.random.rand(config.city_num, config.pos_dimension) # 城市个数×2
# 城市距离矩阵
city_dist_mat = build_dist_mat(city_pos_list) # 二维数组[i,j]: i号到j号的距离

print(city_pos_list)
print(city_dist_mat)

# 遗传算法运行
result,fitness_list = solveTSP(city_dist_mat, True, nIterations=400)
result_pos_list = city_pos_list[result, :]

# 绘图
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# fig = plt.figure()
plt.plot(result_pos_list[:, 0], result_pos_list[:, 1], 'o-r')
plt.title(u"路线")
plt.legend()
# fig.show()
plt.show()

# fig = plt.figure()
plt.plot(fitness_list)
plt.title(u"适应度曲线")
plt.legend()
# fig.show()
plt.show()
