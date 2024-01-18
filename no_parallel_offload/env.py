"""
-*- coding: utf-8 -*-
@author: Zijia Zhao
"""
from other import *
import pandas as pd


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
        df_pht[str(i)] = func1(100, 20)
    # print(df_pht)
    # print(df_pht.sum(axis=1).tolist())
    list_pht = df_pht.sum(axis=1).tolist()
    arr = np.array(list_pht)

    popularity = arr.argsort()[-5:][::-1].tolist()
    return popularity


class ENV:
    def __init__(self, num_app_rsu1, num_app_rsu2, num_task, fn_bs, fn_app, delay_constraints, cache_capacity):
        self.progress = None
        # 初始化数据
        self.num_app_rsu1 = num_app_rsu1
        self.num_app_rsu2 = num_app_rsu2
        self.num_task = num_task
        self.rsu1, self.rsu2 = generate_rsus(num_app_rsu1, num_app_rsu2, self.num_task, fn_bs, fn_app,
                                             delay_constraints, cache_capacity)
        # dn:任务的输入数据大小
        self.dn = [0] * (self.num_app_rsu1 + self.num_app_rsu2) * self.num_task
        # cn:完成任务所需的cpu周期量
        self.cn = [0] * (self.num_app_rsu1 + self.num_app_rsu2) * self.num_task

        self.cpu_remain_app = [0] * (self.num_app_rsu1 + self.num_app_rsu2)
        self.cpu_remain_rsu = [0] * 2

        self.count_wrong = 0
        self.done = False
        self.t_offload = 0
        self.reward = 0
        self.i_task = 0
        self.rn = 5e8
        self.popularity = []
        self.popularity_slice1 = []
        self.popularity_slice2 = []
        self.popularity_state = [0] * 18
        self.t_all = 0
        self.app_num = 0
        self.rsu_num = 0
        self.number_now_task = 0
        self.est = 0
        self.eft = 0
        self.exet = 0
        self.rsu = self.rsu1
        self.rsu_num = 0
        self.local_success = 0
        self.local_unsuccess = 0
        self.offload_success = 0
        self.offload_unsuccess = 0
        self.offload_huancun = 0

    def get_init_state(self, fn_bs, fn_app, delay_constraints, cache_capacity):
        self.count_wrong = 0
        self.t_offload = 0
        self.t_all = 0
        self.app_num = 0
        self.rsu_num = 0
        self.popularity = []
        self.num_app_rsu1 = self.num_app_rsu1
        self.num_app_rsu2 = self.num_app_rsu2
        self.number_now_task = 0
        self.i_task = 0
        self.done = False
        self.rsu1, self.rsu2 = generate_rsus(self.num_app_rsu1, self.num_app_rsu2, self.num_task, fn_bs, fn_app,
                                             delay_constraints, cache_capacity)
        self.popularity_slice1 = self.rsu1.popularity_slice
        self.popularity_slice2 = self.rsu2.popularity_slice
        self.est = 0
        self.eft = 0
        self.exet = 0
        self.rsu = self.rsu1
        self.rsu_num = 0
        self.popularity = self.popularity_slice1
        self.local_success = 0
        self.local_unsuccess = 0
        self.offload_success = 0
        self.offload_unsuccess = 0
        self.offload_huancun = 0

        flattened_list1 = [element for sublist in self.popularity_slice1 for element in sublist]
        flattened_list2 = [element for sublist in self.popularity_slice2 for element in sublist]
        flattened_list = flattened_list1 + flattened_list2
        self.popularity_state[:len(flattened_list)] = flattened_list

        # dn:任务的输入数据大小
        self.dn = [0] * (self.num_app_rsu1 + self.num_app_rsu2) * self.num_task
        # cn:完成任务所需的cpu周期量
        self.cn = [0] * (self.num_app_rsu1 + self.num_app_rsu2) * self.num_task

        # cpu剩余量
        self.cpu_remain_rsu[0] = self.rsu1.cpu_frequency
        self.cpu_remain_rsu[1] = self.rsu2.cpu_frequency
        i = 0
        for app in self.rsu1.apps:
            self.cpu_remain_app[i] = app.cpu_frequency
            i += 1
        for app in self.rsu2.apps:
            self.cpu_remain_app[i] = app.cpu_frequency
            i += 1
        # 进度
        self.progress = [0] * (self.num_app_rsu1 + self.num_app_rsu2) * self.num_task

        state = np.concatenate(
            (self.cn, self.progress, self.cpu_remain_app, self.cpu_remain_rsu, self.popularity_state))
        return state

    def step(self, action):
        # action:是否卸载
        if action[0] > 1:
            action[0] = 1
        if action[0] < -1:
            action[0] = -1
        get1 = 1 if action[0] > 0 else 0  # 是否卸载

        if self.app_num >= self.num_app_rsu1:
            self.app_num = 0
            self.rsu = self.rsu2
            self.rsu_num = 1
            self.popularity = self.popularity_slice2
        app = self.rsu.apps[self.app_num]
        # self.num_task个任务的第几个
        task = app.tasks[self.number_now_task]

        # 当前task所属的片
        slice = task.slice
        T = app.tasks[0].delay_constraints
        Cpu_task = task.cn
        max_before = 0
        # 本地卸载
        if get1 == 0:
            if task.slice == 1:
                self.eft = Cpu_task / app.fn
                self.exet = Cpu_task / app.fn
            else:
                self.exet = Cpu_task / app.fn
                # for ta in app.tasks:
                #     if ta.id in
                for ta in app.tasks:
                    if ta.id in task.predecessors:
                        max_before = max(ta.early_end_time + ta.dn / self.rn, max_before)
                # 最早完成时间
                self.eft = self.exet + max_before
            # 成功的情况：时延小于时延约束 所需cpu小于该车的cpu量
            if self.exet <= T and Cpu_task <= self.cpu_remain_app[self.app_num]:
                self.local_success += 1
                self.dn[self.i_task] = task.dn
                self.cn[self.i_task] = task.cn
                self.progress[self.i_task] = 1
                self.done = True if sum(self.progress) == (
                        self.num_app_rsu1 + self.num_app_rsu2) * self.num_task else False
                # cpu变动
                self.cpu_remain_app[self.app_num] = self.cpu_remain_app[self.app_num] - Cpu_task
                self.reward = - self.exet
                self.t_all += self.exet
                if self.rsu_num == 0:
                    self.rsu1.apps[self.app_num].tasks[self.number_now_task].early_start_time = max_before
                    self.rsu1.apps[self.app_num].tasks[self.number_now_task].early_end_time = self.eft
                else:
                    self.rsu2.apps[self.app_num].tasks[self.number_now_task].early_start_time = max_before
                    self.rsu2.apps[self.app_num].tasks[self.number_now_task].early_end_time = self.eft
            else:
                # 本地卸载失败
                self.local_unsuccess += 1
                self.dn[self.i_task] = task.dn
                self.cn[self.i_task] = task.cn
                self.progress[self.i_task] = 1
                self.done = True if sum(self.progress) == (
                        self.num_app_rsu1 + self.num_app_rsu2) * self.num_task else False
                self.count_wrong += 1
                self.reward = -1 * T
                self.t_all += T
                if self.rsu_num == 0:
                    self.rsu1.apps[self.app_num].tasks[self.number_now_task].early_start_time = max_before
                    self.rsu1.apps[self.app_num].tasks[self.number_now_task].early_end_time = T + max_before
                else:
                    self.rsu2.apps[self.app_num].tasks[self.number_now_task].early_start_time = max_before
                    self.rsu2.apps[self.app_num].tasks[self.number_now_task].early_end_time = T + max_before

        # 卸载情况
        else:
            # 判断该任务是否缓存
            if task.id in self.popularity[slice - 1]:
                self.offload_huancun += 1
                self.exet = 0
                self.dn[self.i_task] = task.dn
                self.cn[self.i_task] = task.cn
                self.progress[self.i_task] = 1
                self.done = True if sum(self.progress) == (
                        self.num_app_rsu1 + self.num_app_rsu2) * self.num_task else False
                self.reward = -0.2
                max_before = 0
                for ta in app.tasks:
                    if ta.id in task.predecessors:
                        max_before = max(ta.early_end_time + ta.dn / self.rn, max_before)
                if self.rsu_num == 0:
                    self.rsu1.apps[self.app_num].tasks[self.number_now_task].early_start_time = max_before
                    self.rsu1.apps[self.app_num].tasks[self.number_now_task].early_end_time = max_before + self.exet
                else:
                    self.rsu2.apps[self.app_num].tasks[self.number_now_task].early_start_time = max_before
                    self.rsu2.apps[self.app_num].tasks[self.number_now_task].early_end_time = max_before + self.exet
            # 未缓存
            else:
                if task.slice == 1:
                    self.exet = Cpu_task / self.rsu.fn_bs
                    self.eft = Cpu_task / self.rsu.fn_bs
                else:
                    self.exet = Cpu_task / self.rsu.fn_bs
                    max_before = 0
                    for predecessor in task.predecessors:
                        for ta in app.tasks:
                            if ta.id == predecessor:
                                max_before = max(ta.early_start_time + ta.dn / self.rn, max_before)
                    # 最早完成时间
                    self.eft = self.exet + max_before
                # 卸载成功
                if Cpu_task <= self.cpu_remain_rsu[self.rsu_num] and self.exet <= T:
                    self.offload_success += 1
                    self.progress[self.i_task] = 1
                    self.done = True if sum(self.progress) == (
                            self.num_app_rsu1 + self.num_app_rsu2) * self.num_task else False
                    self.reward = -1 * self.exet
                    self.dn[self.i_task] = task.dn
                    self.cn[self.i_task] = task.cn
                    self.t_all += self.exet
                    # 卸载完成后，rsu所剩cpu
                    self.cpu_remain_rsu[self.rsu_num] = self.cpu_remain_rsu[self.rsu_num] - Cpu_task
                    # 初始化最早开始时间、最早完成时间
                    if self.rsu_num == 0:
                        self.rsu1.apps[self.app_num].tasks[self.number_now_task].early_start_time = max_before
                        self.rsu1.apps[self.app_num].tasks[self.number_now_task].early_end_time = self.eft
                    else:
                        self.rsu2.apps[self.app_num].tasks[self.number_now_task].early_start_time = max_before
                        self.rsu2.apps[self.app_num].tasks[self.number_now_task].early_end_time = self.eft
                # 卸载失败
                else:
                    self.offload_unsuccess += 1
                    self.progress[self.i_task] = 1
                    self.done = True if sum(self.progress) == (
                            self.num_app_rsu1 + self.num_app_rsu2) * self.num_task else False
                    self.reward = -1 * T
                    self.dn[self.i_task] = task.dn
                    self.cn[self.i_task] = task.cn
                    self.count_wrong += 1
                    self.t_all += T
                    if self.rsu_num == 0:
                        self.rsu1.apps[self.app_num].tasks[self.number_now_task].early_start_time = max_before
                        self.rsu1.apps[self.app_num].tasks[self.number_now_task].early_end_time = max_before + T
                    else:
                        self.rsu2.apps[self.app_num].tasks[self.number_now_task].early_start_time = max_before
                        self.rsu2.apps[self.app_num].tasks[self.number_now_task].early_end_time = max_before + T
        # 进行参数逻辑处理
        if self.number_now_task + 1 == self.num_task:
            self.number_now_task = 0
            self.app_num += 1
        else:
            self.number_now_task += 1

        self.i_task += 1
        state = np.concatenate(
            (self.cn, self.progress, self.cpu_remain_app, self.cpu_remain_rsu, self.popularity_state))
        return state, self.reward, self.done
