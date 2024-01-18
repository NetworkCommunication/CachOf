# CachOf

## Project Introduction

To enhance the efficiency of cache and allocate computing resources, we devise a subtask priority computing approach.
Furthermore, we introduce a dynamic caching solution crafted to meet the requirements of dependent tasks. It can aid in offloading decisions and facilitate load balancing.
Lastly, we propose an offloading strategy based on DRL to intelligently distribute resources.
Additionally, many comparative experiments were carried out in this project. We compare it with alternative algorithms like DQN and GA.

## Running Environment

The running environment depends on the following:

networkx == 3.1，

numpy == 1.26.2，

matplotlib == 3.8.0，

pandas == 2.1.4，

gym == 0.18.0. 

os，

random，

math，

argparse

## Run Ways

Here are five folders corresponding to five sets of experiments, run **run.py** in different folders to easily run the code

## Catalog Structure

### ddpg

Employing the DDPG algorithm for offloading dependent subtasks with the aim of minimizing latency.

**File Introduction:**

- The DAGs_Generator.py is responsible for dynamically generating DAG diagrams.
- The env.py is the code for initializing the environment of DDPG and updating the status and rewards after making decisions.
- The network.py is the network structure of the DDPG algorithm.
- The other.py is mainly used to create and manage Tasks, Apps, and RSUs, which together constitute an important component of the simulation environment.
- The excel_files are used to store experimental results.
- The run.py is the file where the algorithm is run. Run.py also includes other comparative experiments, such as time consumption, average latency, and uninstallation success rate.

### dqn

The DQN algorithm is employed to demonstrate that in a continuous action space, the DQN algorithm is not as effective as the DDPG algorithm.

**File Introduction:**

- The DAGs_Generator.py is responsible for dynamically generating DAG diagrams.
- The dqn.py is a file after parameter changes to DQN.
- The env.py in the file is the environmental configuration of the DQN algorithm.
- The other.py is mainly used to create and manage Tasks, Apps, and RSUs, which together constitute an important component of the simulation environment.
- The excel_files are used to store experimental results.
- The run.py is main run code. Run.py also includes other comparative experiments, such as time consumption, average latency, and uninstallation success rate.

### no_cache

The approach still adopts a static caching method, thereby demonstrating the advantages of the dynamic caching solution we proposed.

**File Introduction:**

- The DAGs_Generator.py is responsible for dynamically generating DAG diagrams.
- The env.py is the code for initializing the environment of DDPG and updating the status and rewards after making decisions.
- The network.py is the network structure of the DDPG algorithm.
- The other.py is mainly used to create and manage Tasks, Apps, and RSUs, which together constitute an important component of the simulation environment.
- The excel_files are used to store experimental results.
- The run.py is main run code. Run.py also includes other comparative experiments, such as time consumption, average latency, and uninstallation success rate.

### no_parallel_offload

The scheme does not involve parallel offloading among subtasks and lacks competition for RSU resources from different app-originated subtasks. This highlights the benefits of our proposed task priority computation.

**File Introduction:**

- The DAGs_Generator.py is responsible for dynamically generating DAG diagrams.
- The env.py is the code for initializing the environment of DDPG and updating the status and rewards after making decisions.
- The network.py is the network structure of the DDPG algorithm.
- The other.py is mainly used to create and manage Tasks, Apps, and RSUs, which together constitute an important component of the simulation environment.
- The excel_files are used to store experimental results.
- The run.py is main run code. Run.py also includes other comparative experiments, such as time consumption, average latency, and uninstallation success rate.

### genetic

The approach employs a genetic algorithm for offloading decisions, demonstrating its inferiority compared to the DDPG reinforcement learning algorithm.

**File Introduction:**

- The DAGs_Generator.py is responsible for dynamically generating DAG diagrams.
- The env.py and its methods provide a simulation environment for genetic algorithms to evaluate, select, and evolve individuals in a population.
- The GA.py is the network structure of the genetic algorithm.
- The other.py is mainly used to create and manage Tasks, Apps, and RSUs, which together constitute an important component of the simulation environment.
- The excel_files are used to store experimental results.
- The run.py is main run code. Run_ra.py also includes other comparative experiments, such as time consumption, average latency, and uninstallation success rate.
