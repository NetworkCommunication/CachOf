"""
@File: run_ga.py
@author: Zijia Zhao
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from genetic.GA import Population
from env import ENV


def run(num_app_rsu1, num_app_rsu2, num_task, fn_bs, fn_app, delay_constraints, cache_capacity):
    def get_fitness(individual, env):
        state = env.get_init_state(fn_bs, fn_app, delay_constraints, cache_capacity)
        total_reward = 0
        done = False
        action_idx = 0

        while not done and action_idx < len(individual.actions):
            action = [individual.actions[action_idx]]
            state, reward, done = env.step(action)
            total_reward += reward
            action_idx += 1

        return total_reward

    # 初始化环境和遗传算法种群
    num_app_rsu1 = num_app_rsu1
    num_app_rsu2 = num_app_rsu2
    num_task = num_task
    env = ENV(num_app_rsu1, num_app_rsu2, num_task, fn_bs, fn_app, delay_constraints, cache_capacity)
    population_size = 50
    # episode
    num_generations = 800
    action_size = (num_app_rsu1 + num_app_rsu2) * num_task  # 假设每个个体将执行10个动作
    action_bound = 1
    score = 0
    population = Population(population_size, action_size, action_bound)
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
    # 遗传算法训练循环
    for gen in range(num_generations):
        score = 0
        population.evolve(lambda ind: get_fitness(ind, env))
        best_individual = max(population.individuals, key=lambda ind: get_fitness(ind, env))
        score = get_fitness(best_individual, env)
        episode_record.append(gen)
        # cost_record.append(score)
        # time_record.append(env.t_all/env.num_task*(env.num_app_rsu1+env.num_app_rsu2))
        # average_time_record.append(env.t_all/(env.num_task*(env.num_app_rsu1+env.num_app_rsu2)))
        rate_record.append((env.local_success + env.offload_success + env.offload_huancun) / (
                env.local_success + env.offload_success + env.offload_huancun + env.local_unsuccess + env.offload_unsuccess))
        # print(f"Generation: {gen}, Best Fitness: {get_fitness(best_individual, env)}")
        print('episode ', gen, 'score %.2f' % score, "    wrong: ", env.count_wrong)
        if gen % 50 == 0:
            episode_record_step.append(gen)
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
    # genetic
    # 变更的参数：APP数量/子任务数量/RSU计算能力fn/APP计算能力fn/时延约束/缓存容量
    num_app_rsu1 = 4
    num_app_rsu2 = 4
    num_task = 7
    fn_bs = 0.6e8
    fn_app = 0.27e8
    delay_constraints = 1.5
    cache_capacity = 7
    run(num_app_rsu1, num_app_rsu2, num_task, fn_bs, fn_app, delay_constraints, cache_capacity)
