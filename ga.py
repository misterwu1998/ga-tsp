import random
import numpy as np

def copy_list(old_arr: [int]):
    new_arr = []
    for element in old_arr:
        new_arr.append(element)
    return new_arr


# 个体类
class Individual:
    def __init__(self, gene_len,distMat, genes=None):
        # 随机生成序列
        if genes is None:
            genes = [i for i in range(gene_len)]
            random.shuffle(genes)
        self.gene_len = gene_len
        self.genes = genes
        self.city_dist_mat = distMat
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        # 计算个体适应度
        fitness = 0.0
        for i in range(self.gene_len - 1):
            # 起始城市和目标城市
            from_idx = self.genes[i]
            to_idx = self.genes[i + 1]
            fitness += self.city_dist_mat[from_idx, to_idx]
        # 连接首尾
        fitness += self.city_dist_mat[self.genes[-1], self.genes[0]]
        return fitness


class Ga:
    def __init__(self, nIndividuals,input_,mutationProbability):
        self.city_dist_mat = input_
        self.best = None  # 每一代的最佳个体
        self.individual_list = []  # 每一代的个体列表
        self.result_list = []  # 每一代对应的解
        self.fitness_list = []  # 每一代对应的适应度
        self.gene_len = np.shape(input_)[0]
        self.individual_num = nIndividuals
        self.mutate_prob = mutationProbability

    def cross(self):
        new_gen = []
        random.shuffle(self.individual_list)
        for i in range(0, self.individual_num - 1, 2):
            # 父代基因
            genes1 = copy_list(self.individual_list[i].genes)
            genes2 = copy_list(self.individual_list[i + 1].genes)
            index1 = random.randint(0, self.gene_len - 2)
            index2 = random.randint(index1, self.gene_len - 1)
            pos1_recorder = {value: idx for idx, value in enumerate(genes1)}
            pos2_recorder = {value: idx for idx, value in enumerate(genes2)}
            # 交叉
            for j in range(index1, index2):
                value1, value2 = genes1[j], genes2[j]
                pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
                genes1[j], genes1[pos1] = genes1[pos1], genes1[j]
                genes2[j], genes2[pos2] = genes2[pos2], genes2[j]
                pos1_recorder[value1], pos1_recorder[value2] = pos1, j
                pos2_recorder[value1], pos2_recorder[value2] = j, pos2
            new_gen.append(Individual(self.gene_len,self.city_dist_mat,genes1))
            new_gen.append(Individual(self.gene_len,self.city_dist_mat,genes2))
        return new_gen

    def mutate(self, new_gen):
        for individual in new_gen:
            if random.random() < self.mutate_prob:
                # 翻转切片
                old_genes = copy_list(individual.genes)
                index1 = random.randint(0, self.gene_len - 2)
                index2 = random.randint(index1, self.gene_len - 1)
                genes_mutate = old_genes[index1:index2]
                genes_mutate.reverse()
                individual.genes = old_genes[:index1] + genes_mutate + old_genes[index2:]
        # 两代合并
        self.individual_list += new_gen

    def select(self):
        # 锦标赛
        group_num = 10  # 小组数
        group_size = 10  # 每小组人数
        group_winner = self.individual_num // group_num  # 每小组获胜人数
        winners = []  # 锦标赛结果
        for i in range(group_num):
            group = []
            for j in range(group_size):
                # 随机组成小组
                player = random.choice(self.individual_list)
                player = Individual(self.gene_len,self.city_dist_mat,player.genes)
                group.append(player)
            group = Ga.rank(group)
            # 取出获胜者
            winners += group[:group_winner]
        self.individual_list = winners

    @staticmethod
    def rank(group):
        # 冒泡排序
        for i in range(1, len(group)):
            for j in range(0, len(group) - i):
                if group[j].fitness > group[j + 1].fitness:
                    group[j], group[j + 1] = group[j + 1], group[j]
        return group

    def next_gen(self):
        # 交叉
        new_gen = self.cross()
        # 变异
        self.mutate(new_gen)
        # 选择
        self.select()
        # 获得这一代的结果
        for individual in self.individual_list:
            if individual.fitness < self.best.fitness:
                self.best = individual

    def train(self, nIterations, initialGene=None):
        '''
            initialGene: 初始基因；具体到当前TSP问题，就是提供一个依次存储了各城市编号的list作为迭代起始状态
        '''
        # 初代种群
        self.individual_list = [Individual(self.gene_len,self.city_dist_mat,initialGene) for _ in range(self.individual_num)]
        self.best = self.individual_list[0]
        # 迭代
        for i in range(nIterations):
            self.next_gen()
            # 连接首尾
            result = copy_list(self.best.genes)
            result.append(result[0])
            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)
        return self.result_list, self.fitness_list

def solveTSP(
    distMat
    , closed=True
    , nIndividuals=50
    , nIterations=500
    , mutationProbability=0.25
    , initialPath = None):
    '''
        distMat[i,j]: i号到j号的距离
        closed: 要闭环的结果吗？
        return: 路径上各点序号（对于闭环的结果，终点就是起点）[], 适应度曲线[]
        initialPath: 提供一个依次存储了各城市编号的list作为迭代起始状态
    '''
    model = Ga(nIndividuals,distMat,mutationProbability)
    ret = model.train(nIterations,initialPath)
    result = list(ret[0][-1]) # 终点就是起点
    fitness_list = ret[1]
    if not closed: # 不想要闭环的路径
        # 那就需要在闭环路径上找到最长的一个弧，从它切开
        i = np.array(result[:-1])
        j = np.array(result[1:])
        stepLengths = distMat[i,j] # [x]: 闭环路径上第x位到第x+1位这段弧的长度（注意不是x号点到x+1号点）
        ndxOfLongestStep = np.argmax(stepLengths) # 是一维数组，不需要unravel_index
        # 从原本的第x到第x+1步这里切断
        openedResult = []
        for x in range(ndxOfLongestStep+1, len(result)): # 切口的右侧
            openedResult.append(result[x])
        for x in range(ndxOfLongestStep+1): # 切口的左侧
            openedResult.append(result[x])
        result = openedResult
    # 不想要闭环的路径
    return result, fitness_list
