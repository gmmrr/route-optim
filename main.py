import os, sys

from models import environment
from models import agent
from models import dijkstra

def sumo_config():
    # os.environ["SUMO_HOME"] = '$SUMO_HOME' # -- change to your path to $SUMO_HOME if necessary

    # Check if SUMO sucessfully configured
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")


if __name__ == '__main__':

    # 01 Setup SUMO
    sumo_config()


    # 02 Configure network variables
    network_file = './network_files/ncku_network.net.xml'
    congestion = []  # can be defined, but if it is empty, env will randomly decide congested edges
    traffic_light = [
        ["1725808117", 240]
    ]
    start_node = "864831599"
    end_node = "5739293224"


    # 03 Initiate Environment
    env = environment.traffic_env(
        network_file = network_file,
        congestion = congestion,
        traffic_light = traffic_light,
        evaluation = "time", # Type: "destination" | "time"
        congestion_level = "low",  # Type: "low" | "medium" | "high", only applied if the congestion is not defined
    )
    num_episodes = 5000
    num_converge = 5


    # 04 Activate Agent
    # -------------------
    # Dijkstra Algorithm
    # -------------------
    print(f"\nDijkstra Algorithm{'.' * 100}")
    Dijkstra = dijkstra.Dijkstra(env, start_node, end_node)
    node_path, edge_path = Dijkstra.search()
    env.plot_visualised_result(edge_path)

    # -------------------
    # Q_Learning Algorithm
    # -------------------
    print(f"\nQ_Learning Algorithm{'.' * 100}")
    QLearning_agent = agent.Q_Learning(env, start_node, end_node)
    node_path, edge_path, episode, logs = QLearning_agent.train(num_episodes, num_converge)
    env.plot_performance(episode, logs)
    env.plot_visualised_result(edge_path)

    # -------------------
    # SARSA Algorithm
    # -------------------
    print(f"\nSARSA Algorithm{'.' * 100}")
    SARSA_agent = agent.SARSA(env, start_node, end_node, exploration_rate = 0.1)
    node_path, edge_path, episode, logs = SARSA_agent.train(num_episodes, num_converge)
    env.plot_performance(episode, logs)
    env.plot_visualised_result(edge_path)
