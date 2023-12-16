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
    # -------------------
    # 2x3 Traffic Network, for testing
    # [A, B, C, D, E, F, G, H, I, J, K, L, M, N]
    # -------------------
    # network_file = './network_files/2x3_network.net.xml'
    # congestion = [("gneF_I", 10), ("gneI_F", 10), ("gneB_E", 20), ("gneE_B", 20), ("gneJ_M", 30), ("gneM_J", 30)]
    # traffic_light = [("B", 5), ("I", 5), ("G", 5)]
    # start_node = "A"
    # end_node = "N"

    # -------------------
    # Sunway City Traffic Network
    # [101: Sunway University, 102: Monash University,  103: Sunway Geo,
    #  104: Sunway Medical,    105: Taylors University, 106: Sunway Pyramid,
    #  107: Sunway Lagoon,     108: PJS10 Park]
    # -------------------
    network_file = './network_files/sunway_network.net.xml'
    congestion = [("gne2124969573_1000000001", 10), ("gne677583745_2302498575", 10), ("gne5236931684_143675326", 20), ("gne1000000001_5735834058", 20), ("gne10734244602_1640449323", 10)]
    traffic_light = [(["2124969573", "2124969571"], 5), (["677583896", "1670458823"], 5), (["2210133573", "2210133562", "2210133501", "2210133223"], 5), (["4123498067", "4123498068", "2210132568", "2210132847"], 5), (["1197884608", "1197880914", "1197884584", "269953766"], 5), (["5762726921", "8948947765", "10845806303", "10845816012"], 5), (["677583804", "677583801", "677583803", "677583802"], 5), (["7211376203", "7211376202", "7211376200", "7211376201"], 5), (["2747527085", "1636307448", "678457498", "5780613945", "5780613944"], 5), (["5727497437", "5727497436", "678457587", "678457535"], 5), (["463099148", "1197913517"], 5), ("712814465", 5), ("1197913486", 5), ("9209244285", 5)]
    start_node = "101"
    end_node = "105"

    # 03 Initiate Environment
    env = environment.traffic_env(
        network_file = network_file,
        congestion = congestion,
        traffic_light = traffic_light,
        evaluation = "time", # Type: "destination" | "time"
        congestion_level = "low",  # Type: "low" | "medium" | "high", only applied if the congestion is not defined
        travel_speed = 80  # Type: number
    )
    num_episodes = 5000
    num_converge = 5


    # 04 Activate Agent
    # -------------------
    # Dijkstra Algorithm, as a reference
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
