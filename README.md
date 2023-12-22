# Routing Optimization in Traffic Network using Reinforcement Learning

## Project Scope

As there are multiple factors involved in selecting the most optimal route, below are the factors that have been preset for this study:
- Traffic network is not updated in real-time
- Vehicle Speed is constant at 80 km/hr

## Method of Evaluation

1. Comparision of the routes selected from the agent vs the baseline model (Dijkstra) 
2. Evaluate if the models managed to converge
3. Comparision of the number of episodes taken to converge (SARSA vs Q_Learning)
4. Comparision of the time taken for the computation

## Setup 

1. Download SUMO (https://sumo.dlr.de/docs/Downloads.php)
2. Clone this repository to your local machine
3. Install the necessary packages by following operations:
```python
pip3 install -r requirements.txt
```
4. Update the main.py with your SUMO directory to set the environment variable
```python
def sumo_config():
    os.environ["SUMO_HOME"] = '$SUMO_HOME' # -- change to your path to $SUMO_HOME
    ...
```
5. Upload your netedit file and update the network_file variable
```python
network_file = './network_files/2x3_network.net.xml'
```
**More on Netedit:** https://sumo.dlr.de/docs/Netedit/index.html 

6. Edit following parameters as part of environment
```python
env = environment.traffic_env(
    network_file = network_file,
    congestion = congestion,
    traffic_light = traffic_light,
    evaluation = "time", # Type: "destination" | "time"
    congestion_level = "low",  # Type: "low" | "medium" | "high", only applied if the congestion is not defined
    travel_speed = 80  # Type: number
)
```
7. Run the code
```terminal
$ python3 main.py
```

## Test Cases

### Test Case 1 - Ideal Reward Function

### Test Case 2 - Traffic Density

## Graph Plotting

1. **Route Map**: In `main.py`, the function below maps the routes produced. 
```python
env.visualise_plot(edge_path)
```

2. **Performance Plot**: In `main.py`, the function below creates a line plot on the performance of each episode. This is also the learning curve of the model.
```python
env.plot_performance(number_of_episode, logs)
```


RL的終止條件是如果5次次數一樣
