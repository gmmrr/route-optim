import os, sys
import xml.etree.ElementTree as ET

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


def tls_from_tllxml(file_name):
    """
    Make tls data from tll.xml

    Args:
    - file_name: tll.xml file name
        <additionals>
            <tlLogic id="10726190730" type="static" programID="0" offset="0">
                <phase duration="42" state="GGgrrrGGgrrr"/>
                <phase duration="3"  state="yyyrrryyyrrr"/>
                <phase duration="42" state="rrrGGgrrrGGg"/>
                <phase duration="3"  state="rrryyyrrryyy"/>
            </tlLogic>
            <tlLogic id="" type="" programID="" offset="">
                <phase/>
            </tlLogic>
            ...
        </additionals>

    Returns:
    - tls_data: dictionary of tls data
        tls_data[tl_id][link_index] = [90]
    """
    tree = ET.parse(file_name)
    root = tree.getroot()

    tls_data = {}
    for tl in root.findall('.//tlLogic'):
        tl_id = tl.get('id')

        if tl_id not in tls_data:
            tls_data[tl_id] = {}
        else:
            sys.exit(f"Error: {tl_id} duplicated")

        phases = tl.findall('.//phase')

        for phase in phases:
            duration = phase.get('duration')
            state = phase.get('state')

            # Make tls_data[tl_id][link_index] = nth char in state repeating duration times
            for link_index in range(len(state)):
                if link_index not in tls_data[tl_id]:
                    tls_data[tl_id][link_index] = []
                tls_data[tl_id][link_index] += [state[link_index] for _ in range(int(duration))]

    return tls_data


if __name__ == '__main__':

    # 01 Setup SUMO
    sumo_config()


    # 02 Configure network variables
    network_file = './network_files/ncku_network.net.xml'  # download .osm and then netconvert to .net.xml, see more in "https://www.openstreetmap.org/" and "config.txt" is the netconvert config
    tls = tls_from_tllxml('./network_files/ncku_network.tll.xml')  # tll.xml is exported from netedit
    congestion = []  # can be defined, but if it is empty, env will randomly decide congested edges
    start_node = "864831599"  # can be defined, the scope is the nodes in the network
    end_node = "5739293224"

    # 03 Initiate Environment
    env = environment.traffic_env(
        network_file = network_file,
        tls = tls,
        congestion = congestion,
        evaluation = "time", # Type: "destination" | "time"
        congestion_level = "low",  # Type: "low" | "medium" | "high", only applied if the congestion is not defined
    )


    # 04 Activate Agent
    # -------------------
    # Dijkstra Algorithm
    # -------------------
    # print(f"\nDijkstra Algorithm{'.' * 100}")
    # Dijkstra = dijkstra.Dijkstra(env, start_node, end_node)
    # node_path, edge_path = Dijkstra.search()
    # env.plot_visualised_result(edge_path)

    # -------------------
    # Q_Learning Algorithm
    # -------------------
    # print(f"\nQ_Learning Algorithm{'.' * 100}")
    # QLearning_agent = agent.Q_Learning(env, start_node, end_node)
    # node_path, edge_path, episode, logs = QLearning_agent.train(5000, 5)  # limit of episodes, threshold to converge
    # env.plot_performance(episode, logs)
    # env.plot_visualised_result(edge_path)

    # -------------------
    # SARSA Algorithm
    # -------------------
    print(f"\nSARSA Algorithm{'.' * 100}")
    SARSA_agent = agent.SARSA(env, start_node, end_node)
    node_path, edge_path, episode, logs = SARSA_agent.train(5000, 20)  # limit of episodes, threshold to converge
    env.plot_performance(episode, logs)
    env.plot_visualised_result(edge_path)
