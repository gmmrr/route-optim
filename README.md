# route-optim
<img width="764" alt="Screenshot 2023-12-24 at 22 18 03" src="https://github.com/gmmrr/route-optim/assets/88712124/1701dcbb-8077-4cc1-975c-46dd5f7bc313">



## Description
This project aims to solve the route optimisation problem of individual vehicle by reinforcement learning.<br>

Giving a set of start terminal and end terminal, it can assign a particular vehicle to deal with it intuitively, and it will follow the route computed by RL.<br>

For multiple vehicles and demands case, <a href="https://github.com/gmmrr/fleet-route-optim">gmmrr/fleet-route-optim<a/>  is another version aims to deal with vehicles in fleet.<br>

This repo is part of Guanming's capstone project.

## Result
<img width="764" alt="Screenshot 2023-12-24 at 22 15 56" src="https://github.com/gmmrr/route-optim/assets/88712124/254ea7c8-cd59-43da-8852-9644f2310d37">

```
Processing Time: 0.09724 seconds
Travelled Time: 6.15 mins
```

<img width="637" alt="Screenshot 2023-12-24 at 22 17 12" src="https://github.com/gmmrr/route-optim/assets/88712124/fe94d8b1-d7aa-4634-acea-c739f13b23ff">
<img width="764" alt="Screenshot 2023-12-24 at 22 18 03" src="https://github.com/gmmrr/route-optim/assets/88712124/1701dcbb-8077-4cc1-975c-46dd5f7bc313">

```
Processing Time: 42.869093 seconds
Travelled Time: 6.67 mins
```

<img width="638" alt="Screenshot 2023-12-24 at 22 22 46" src="https://github.com/gmmrr/route-optim/assets/88712124/04b7ad25-2218-4a7b-892a-8b0ad22a2af5">
<img width="763" alt="Screenshot 2023-12-24 at 22 23 37" src="https://github.com/gmmrr/route-optim/assets/88712124/7d20d602-06a7-46de-9dbd-642e82c9db3a">


```
Processing Time: 161.478861 seconds
Travelled Time: 6.65 mins
```





## Project Scope
1. **Congestion is randomly generated**<br>
    Similar to mentioned situation above. It is randomly chosen from edges space, and it can be defined in ```fleet_environment.py``` to low, medium, or high level.

2. **Speed is a constant**<br>
    Net downloaded from OSM website helps classify the edge type, like primary, secondary, residential highway. Each of them has a defined speed. In this project, we don't take acceleration into consideration. Thus, it seems like to be far away from the practical case.

3. **Traffic light is set in a 90 seconds interval**<br>
    Even if it is close to the practical case, it is still not real. They are set as a program rather than a constant pattern in reality.

4. **The terminal condition of RL**<br>
    It is set that convergence occurs when time taken (round to the second decimal place) in 5 episodes is consistent.

## Setup
1. Download SUMO (https://sumo.dlr.de/docs/Downloads.php)
2. Clone this repository to your local machine
3. Install the necessary packages by following operations:
```python
$ pip3 install -r requirements.txt
```
4. Update the main.py with your SUMO directory to set the environment variable
```python
def sumo_config():
    os.environ["SUMO_HOME"] = '$SUMO_HOME' # -- change to your path to $SUMO_HOME
    ...
```
5. Upload your netedit file and update the network_file variable
```python
network_file = './network_files/ncku_network.net.xml'
```
More on **OSM website**: https://www.openstreetmap.org/ <br>
Config command is saved in ```./network_files/config.txt```

6. Upload your traffic_light file
```python
tls = tls_from_tllxml('./network_files/ncku_network.tll.xml')
```
This file can be converted by **Netedit**, more on https://sumo.dlr.de/docs/Netedit/index.html

7. Edit start_node and end_node in ```main.py```
```python
# 02 Configure network variables
start_node = "864831599"  # can be defined, the scope is the nodes in the network
end_node = "5739293224"
```
8. Run the code
```terminal
$ python3 main.py
```

## Customisable Section
In ```agent.py```, we can set
```python
# Hyperparameters for Q_Learning
learning_rate = 0.9  # alpha
discount_factor = 0.1  # gamma

# Hyperparameters for SARSA
learning_rate = 0.9  # alpha
discount_factor = 0.1  # gamma
exploration_rate = 0.05  # ratio of exploration and exploitation
```
and we have
```python
reward_lst = [-100, -100, -100, 10, 100, -1]
```
They are defined as ```[invalid_action_reward, dead_end_reward, loop_reward, completion_reward, bonus_reward, continue_reward]``` respectively.
