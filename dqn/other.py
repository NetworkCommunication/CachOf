"""
-*- coding: utf-8 -*-
@author: Zijia Zhao
"""
import random
import numpy as np
import pandas as pd
from DAGs_Generator import workflows_generator
from DAGs_level import calculate_uninstall_priority


class RSU:
    def __init__(self, cpu_frequency, fn_bs, apps, popularity_slice, popularity):
        self.cpu_frequency = cpu_frequency
        self.fn_bs = fn_bs
        self.apps = apps
        self.popularity_slice = popularity_slice
        self.popularity = popularity


class App:
    def __init__(self, id, tasks, fn, dict_traget, cpu_frequency):
        self.id = id
        self.tasks = tasks
        self.fn = fn
        self.dict_traget = dict_traget
        self.cpu_frequency = cpu_frequency


class Task:
    # uninstall_priority 卸载优先级
    # late_start_date 最晚开始时间(调度优先级)
    def __init__(self, id, dn, cn, delay_constraints, uninstall_priority, slice, predecessors, early_start_time,
                 early_end_time):
        self.id = id
        self.dn = dn
        self.cn = cn
        self.delay_constraints = delay_constraints
        self.uninstall_priority = uninstall_priority
        self.slice = slice
        self.predecessors = predecessors
        self.early_start_time = early_start_time
        self.early_end_time = early_end_time


def generate_task(new_uninstall_priorities, i, max_value, dict_target, new_edges,delay_constraints):
    keys = list(new_uninstall_priorities.keys())
    task_id = keys[i]
    keys_for_value = get_keys_from_value(dict_target, task_id)
    # 得到原始值，找到原始值的前驱
    predecessors = [edge[0] for edge in new_edges if edge[1] == keys_for_value[0]]
    # 根据原始前驱获得更新前驱
    new_predecessors = []
    for processor in predecessors:
        new_predecessors.append(dict_target[processor])
    uninstall_priority = new_uninstall_priorities[task_id]
    temp_value_1 = max_value / 3
    temp_value_2 = temp_value_1 * 2
    if uninstall_priority <= temp_value_1:
        slice_num = 1
    elif uninstall_priority <= temp_value_2:
        slice_num = 2
    else:
        slice_num = 3
    cn = random.randint(1e7, 0.5e8)
    return Task(task_id, 2 * cn, cn,delay_constraints, uninstall_priority, slice_num, new_predecessors, 0, 0)


def replace(value, dict_target):
    return dict_target[value]


def generate_apps(num_apps, num_task,fn_app,delay_constraints):
    apps = []
    for j in range(num_apps):
        edges, position = workflows_generator(n=(num_task - 2))
        points = set(position)
        new_edges = [list(item) for item in edges]
        list_task_ids = list(range(0, 40))
        # 旧点，新点对应关系
        dict_target = {}
        for point in points:
            target = random.choice(list_task_ids)
            list_task_ids.remove(target)
            dict_target.update({point: target})
        uninstall_priorities = calculate_uninstall_priority(edges)
        # 更新完的点的卸载优先级
        new_uninstall_priorities = {}
        max_vlaue = -1
        for key, value in uninstall_priorities.items():
            new_key = replace(key, dict_target)
            max_vlaue = max(max_vlaue, value)
            new_uninstall_priorities.update({new_key: value})
        tasks = []
        for i in range(num_task):
            task = generate_task(new_uninstall_priorities, i, max_vlaue, dict_target, new_edges,delay_constraints)
            tasks.append(task)

        app = App(j + 1, tasks, fn_app, dict_target, 15e7)
        app.tasks = sorted(app.tasks, key=lambda x: (x.uninstall_priority, app.tasks.index(x)))
        apps.append(app)
    return apps


def get_keys_from_value(dictionary, value):
    keys_list = [key for key, val in dictionary.items() if val == value]
    return keys_list


def get_popularity():
    def func1(amount, num):
        list1 = []
        for i in range(0, num - 1):
            a = random.randint(0, amount)  # 生成 n-1 个随机节点你
            list1.append(a)
        list1.sort()  # 节点排序
        list1.append(amount)  # 设置第 n 个节点为amount，即总金额

        list2 = []
        for i in range(len(list1)):
            if i == 0:
                b = list1[i]  # 第一段长度为第 1 个节点 - 0
            else:
                b = list1[i] - list1[i - 1]  # 其余段为第 n 个节点 - 第 n-1 个节点
            list2.append(b)
        return list2

    df_pht = pd.DataFrame()

    for i in range(3):
        df_pht[str(i)] = func1(100, 40)
    # print(df_pht)
    # print(df_pht.sum(axis=1).tolist())
    list_pht = df_pht.sum(axis=1).tolist()
    arr = np.array(list_pht)

    popularity = arr.argsort()[-5:][::-1].tolist()
    return popularity


def knapsack_01(weights, values, capacity):
    n = len(values)
    # 创建一个二维数组来保存子问题的解决方案
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    selected_items = []

    # 动态规划主循环
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] > w:
                dp[i][w] = dp[i - 1][w]
            else:
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])

    # 回溯找出所选物品
    total_value = dp[n][capacity]
    remaining_capacity = capacity
    for i in range(n, 0, -1):
        if total_value <= 0:
            break
        if total_value == dp[i - 1][remaining_capacity]:
            continue
        else:
            selected_items.append(i - 1)
            total_value -= values[i - 1]
            remaining_capacity -= weights[i - 1]

    selected_items.reverse()
    return [item for item in selected_items]


def get_popularity_slice(rsu_apps, popularity,cache_capacity):
    res = []
    popularity_slice1 = []
    popularity_slice2 = []
    popularity_slice3 = []
    dict_slice1 = {}
    dict_slice2 = {}
    dict_slice3 = {}
    for pop in popularity:
        dict_slice1.update({pop: 0})
        dict_slice2.update({pop: 0})
        dict_slice3.update({pop: 0})
    for app in rsu_apps:
        for task in app.tasks:
            if task.slice == 1:
                if task.id in popularity:
                    dict_slice1.update({task.id: dict_slice1.get(task.id) + 1})
            elif task.slice == 2:
                if task.id in popularity:
                    dict_slice2.update({task.id: dict_slice2.get(task.id) + 1})
            elif task.slice == 3:
                if task.id in popularity:
                    dict_slice3.update({task.id: dict_slice3.get(task.id) + 1})
    values_1 = list(dict_slice1.values())
    values_2 = list(dict_slice2.values())
    values_3 = list(dict_slice3.values())
    weights = [2, 3, 4, 2, 2]
    selected_items_1 = knapsack_01(weights, values_1, cache_capacity)
    selected_items_2 = knapsack_01(weights, values_2, cache_capacity)
    selected_items_3 = knapsack_01(weights, values_3, cache_capacity)
    for i in selected_items_1:
        keys_list = list(dict_slice1.keys())
        popularity_slice1.append(keys_list[i])
    for i in selected_items_2:
        keys_list = list(dict_slice2.keys())
        popularity_slice2.append(keys_list[i])
    for i in selected_items_3:
        keys_list = list(dict_slice3.keys())
        popularity_slice3.append(keys_list[i])
    res.append(popularity_slice1)
    res.append(popularity_slice2)
    res.append(popularity_slice3)
    return res


# Generating RSUs
def generate_rsus(num_app1, num_app2, num_task,fn_bs,fn_app,delay_constraints,cache_capacity):
    popularity = get_popularity()

    rsu1_apps = generate_apps(num_app1, num_task,fn_app,delay_constraints)  # Generating 10 apps for RSU 1
    popularity_slice1 = get_popularity_slice(rsu1_apps, popularity,cache_capacity)
    rsu2_apps = generate_apps(num_app2, num_task,fn_app,delay_constraints)  # Generating 15 apps for RSU 2
    popularity_slice2 = get_popularity_slice(rsu2_apps, popularity,cache_capacity)
    # 1.75e9  0.2e9 random.randint(1e8, 2e8)
    rsu1 = RSU(cpu_frequency=6.5e8, fn_bs=fn_bs, apps=rsu1_apps, popularity_slice=popularity_slice1,
               popularity=popularity)
    rsu2 = RSU(cpu_frequency=6.5e8, fn_bs=fn_bs, apps=rsu2_apps, popularity_slice=popularity_slice2,
               popularity=popularity)
    return rsu1, rsu2

# generate_rsus(15, 10)
