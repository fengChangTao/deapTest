import random
import numpy
import math
import os
import operator
from deap import base, creator, tools, gp, algorithms

# 加载数据集


def load_data(file_path):
    data = {
        'edges': [],
        'costs': [],
        'demands': [],
        'availability': [],
        'deadheading_costs': [],
    }

    with open(file_path, 'r') as f:
        lines = f.readlines()

    read_edges = False

    for line in lines:
        if 'LISTA_ARISTAS_REQ' in line:
            read_edges = True
            continue

        if read_edges:
            if 'DEPOSITO' in line:
                break

            edge_data = line.strip().strip('()').replace(',', '').replace(')', '').split()
            data['edges'].append((int(edge_data[0]), int(edge_data[1])))
            data['costs'].append(int(edge_data[3]))
            data['demands'].append(int(edge_data[5]))
            data['availability'].append(random.uniform(0.8, 1.0))  # 随机生成可用性
            data['deadheading_costs'].append(random.randint(0, 10))  # 随机生成deadheading费用

    return data



data = load_data('gdb1.dat')



# 定义评估函数
def evaluate(individual):
    # 使用加载的数据评估个体
    # 考虑不确定性因素，例如任务存在性、需求、deadheading费用和路径可用性
    fitness = 0

    # 在此示例中，我们将任务存在性表示为随机事件，概率为0.8
    presence_probability = 0.8

    for i, edge in enumerate(data['edges']):
        if random.random() < presence_probability:  # 考虑任务存在性
            if random.random() < data['availability'][i]:  # 考虑路径可用性
                fitness += data['costs'][i] + data['demands'][i] * data['deadheading_costs'][i]  # 计算成本和需求与deadheading费用的乘积之和

    return fitness,


# 创建类型
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# 定义基础原语
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.truediv, 2)  # 添加除法操作
pset.addPrimitive(max, 2)
pset.addPrimitive(min, 2)
pset.addPrimitive(operator.lt, 2)  # 添加小于操作
pset.addPrimitive(operator.le, 2)  # 添加小于等于操作
pset.addPrimitive(operator.gt, 2)  # 添加大于操作
pset.addPrimitive(operator.ge, 2)  # 添加大于等于操作
pset.addPrimitive(operator.eq, 2)  # 添加等于操作
pset.addPrimitive(operator.ne, 2)  # 添加不等于操作
pset.addPrimitive(operator.neg, 1)  # 添加取负操作
pset.addPrimitive(operator.abs, 1)  # 添加绝对值操作
pset.addPrimitive(math.ceil, 1)  # 添加向上取整操作
pset.addPrimitive(math.floor, 1)  # 添加向下取整操作
pset.addPrimitive(math.sqrt, 1)  # 添加平方根操作
pset.addEphemeralConstant("rand101", lambda: random.randint(-10, 10))
pset.renameArguments(ARG0='x')


# 定义遗传算法参数
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# 主要的遗传算法循环
def main():
    random.seed(42)
    pop = toolbox.population(n=150)
    hof = tools.HallOfFame(1)

    # 注册统计数据
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    # 运行遗传算法
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=200, stats=mstats,
                                   halloffame=hof, verbose=True)

    # 返回最佳解决方案
    return hof[0], log

if __name__ == "__main__":
    best_solution, log = main()
    print("Best solution:", best_solution)
