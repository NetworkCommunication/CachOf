"""
-*- coding: utf-8 -*-
@author: Zijia Zhao
"""
from matplotlib import pyplot as plt
import pandas as pd
from env import ENV
from network import Agent
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def run(num_app_rsu1, num_app_rsu2, num_task, fn_bs, fn_app, delay_constraints, cache_capacity):
    num_app_rsu1 = num_app_rsu1
    num_app_rsu2 = num_app_rsu2
    num_task = num_task
    env = ENV(num_app_rsu1, num_app_rsu2, num_task, fn_bs, fn_app, delay_constraints, cache_capacity)
    n_actions = 1
    n_state = (num_app_rsu1 + num_app_rsu2) * num_task * 2 + 2 + num_app_rsu1 + num_app_rsu2 + 18
    MECSnet = Agent(alpha=0.00001, beta=0.0001, input_dims=n_state,
                    tau=0.01, env=env, batch_size=64, layer1_size=300,
                    layer2_size=100, n_actions=n_actions)
    score_record = []
    score_record_step = []
    count_record = []
    count_record_step = []
    episode_record = []
    episode_record_step = []
    cost_record = []
    cost_record_step = []
    time_record = []
    time_record_step = []
    average_time_record = []
    average_time_record_step = []
    rate_record = []
    rate_record_step = []
    for i in range(800):
        score = 0
        if i == 50:
            print("stop")
        obs = env.get_init_state(fn_bs, fn_app, delay_constraints, cache_capacity)
        # print(obs)
        done = False

        while not done:
            act = MECSnet.choose_action(obs)

            new_state, reward, done = env.step(act)
            MECSnet.remember(obs, act, reward, new_state, int(done))
            MECSnet.learn()
            score += reward
            obs = new_state

        episode_record.append(i)
        # cost_record.append(score)
        # time_record.append(env.t_all/env.num_task*(env.num_app_rsu1+env.num_app_rsu2))
        # average_time_record.append(env.t_all/(env.num_task*(env.num_app_rsu1+env.num_app_rsu2)))
        rate_record.append((env.local_success + env.offload_success + env.offload_huancun) / (
                    env.local_success + env.offload_success + env.offload_huancun + env.local_unsuccess + env.offload_unsuccess))
        print('episode ', i, 'score %.2f' % score, "    wrong: ", env.count_wrong)
        # print("local_success:",env.local_success, 'local_unsuccess', env.local_unsuccess," offload_success:", env.offload_success," offload_unnsuccess:", env.offload_unsuccess,"offload_huancun",env.offload_huancun)
        # count_record.append(1 - env.count_wrong / num_task)
        if i % 50 == 0:
            MECSnet.save_models()
            episode_record_step.append(i)
            # cost_record_step.append(np.mean(cost_record))
            # time_record_step.append(np.mean(time_record))
            # average_time_record_step.append(np.mean(average_time_record))
            rate_record_step.append(np.mean(rate_record))

    # df = pd.DataFrame({"Episode": episode_record_step, "Cost": cost_record_step}).set_index('Episode')
    # df = pd.DataFrame({"Episode": episode_record_step, "Time": time_record_step}).set_index('Episode')
    # df = pd.DataFrame({"Episode": episode_record_step, "average_time": average_time_record_step}).set_index('Episode')
    df = pd.DataFrame({"Episode": episode_record_step, "rate_record": rate_record_step}).set_index('Episode')
    df.to_excel("excel_files/episode_SR/5.xlsx")

    plt.figure()
    x_data = range(len(rate_record_step))
    plt.plot(x_data, rate_record_step)

    plt.show()


if __name__ == '__main__':
    # no cache
    # 变更的参数：APP数量/子任务数量/RSU计算能力fn/APP计算能力fn/时延约束/缓存容量
    num_app_rsu1 = 4
    num_app_rsu2 = 4
    num_task = 7
    fn_bs = 0.6e8
    fn_app = 0.27e8
    delay_constraints = 1.5
    cache_capacity = 7
    run(num_app_rsu1, num_app_rsu2, num_task, fn_bs, fn_app, delay_constraints, cache_capacity)
