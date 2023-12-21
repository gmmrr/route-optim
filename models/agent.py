import numpy as np
import sys
import datetime


class rl_agent():
    def __init__ (self, env, start_node, end_node, learning_rate, discount_factor):
        # Define the learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Initialise environment
        self.env = env
        self.env.set_start_end(start_node, end_node)  # let the env knows where are the start and end nodes


    # Reset agent
    def reset(self):
        self.q_table = np.zeros((len(self.env.state_space), len(self.env.action_space)))  # state_space * action_space
        self.logs = {}  # self.logs[episode] = [node_path, edge_path]
        self.best_result = 0


    def act(self):
        pass  # let derived classese define themself


    def step(self, action, node_path, edge_path):
        # 0. initialise step
        is_terminate = False
        current_state = node_path[-1]
        current_edge = edge_path[-1] if edge_path else None

        outgoing_edges = self.env.decode_node_to_edges(current_state, direction = 'outgoing')

        # 1. Determine reward paramaters
        # -------------------
        # START OF EDIT
        # -------------------
        invalid_action_reward = -50
        dead_end_reward = -50
        loop_reward = -30  # = -50
        completion_reward = 50
        bonus_reward = 50  # = ((self.best_result-current_result)/self.best_result)*100 + 50
        continue_reward = 0
        # -------------------
        # END OF EDIT
        # -------------------

        # Define the reward
        reward = continue_reward


        # 2. Compute the reward and the next state
        # Case 1. Out-Of-Bound Action
        if action not in self.env.decode_edges_to_actions(outgoing_edges):  # e.g. it may be unable to turn right
            reward += invalid_action_reward
            next_state = current_state
            next_edge = current_edge

        # Case 2. Valid Action
        else:
            next_edge = self.env.decode_edges_action_to_edge(outgoing_edges, action)
            next_state = self.env.decode_edge_to_node(next_edge, direction = 'end')

            # Case 2-1. End Node
            if next_state in self.env.end_node:
                reward += completion_reward
                is_terminate = True

                # check if the route is the shortest distance/time
                if self.env.evaluation in ("time"):
                    current_result = self.env.get_edge_time(edge_path + [next_edge]) + self.env.get_tl_offset(edge_path + [next_edge])
                else:
                    current_result = self.env.get_edge_distance(edge_path + [next_edge])

                if self.best_result == 0:
                    self.best_result = current_result
                elif current_result < self.best_result:
                    for edge in edge_path:
                        state_index = self.env.state_space.index(self.env.decode_edge_to_node(edge, direction = 'start'))
                        action_index = self.env.edge_label[edge]
                        self.q_table[state_index][action_index] += bonus_reward
                    self.best_result = current_result

            # Case 2-2. Dead-end Route
            elif not self.env.decode_node_to_edges(next_state, direction = 'outgoing'):
                reward += dead_end_reward
                is_terminate = True

                # Backtrack and find bottleneck
                for edge in reversed(edge_path):
                    if len(self.env.decode_node_to_edges(self.env.decode_edge_to_node(edge, direction = 'end'), direction = 'outgoing')) > 1:
                        break

                    state_index = self.env.state_space.index(self.env.decode_edge_to_node(edge, direction = 'start'))
                    action_index = self.env.edge_label[edge]
                    self.q_table[state_index][action_index] += dead_end_reward

            # Case 2-3. Travelling
            elif current_edge != None:
                # Case 2-4. Travelling in a loop
                if (current_edge, next_edge) in [(edge_path[i], edge_path[i+1]) for i in range(len(edge_path)-1)]:
                    reward += loop_reward

        return next_edge, next_state, reward, is_terminate  # return the next state, reward and is_terminate


    # Update the Q-table
    def learn(self, current_state, action, next_state, reward):
        # 1. Get original Q-value
        q_predict = self.q_table[self.env.state_space.index(current_state)][action]

        # 2. Calculate how much Q-value should change
        # ---------------------------------- #
        # Q(S,a) = R + gamma * max(Q(S',a')  #
        # ---------------------------------- #
        q_target = reward + self.discount_factor * np.max(self.q_table[self.env.state_space.index(next_state)])
        # what we need is to find the max one from all q_table[next_state][action]

        # 3. Update Q-value practically
        # -------------------------------------------------------------- #
        # Q(S,a) = Q(S,a) + alpha * (R + gamma * max(Q(S',a') - Q(S,a))  #
        # -------------------------------------------------------------- #
        self.q_table[self.env.state_space.index(current_state)][action] += self.learning_rate * (q_target - q_predict)


    # Main function implemented
    def train(self, num_episodes, threshold):
        start_time = datetime.datetime.now() # record the start time
        self.reset()  # initialise agent
        print('Training Start...')

        # Iterate through episodes
        for episode in range(num_episodes):
            # Initialise state
            node_path = [self.env.start_node]
            edge_path = []
            is_terminate = False

            # Iterate until reach the assigned terminate
            while True:
                last_state = node_path[-1]
                if is_terminate or last_state in self.env.end_node:
                    break

                # Decide the action
                action = self.act(last_state)

                # Take the action and observe the outcome
                next_edge, next_state, reward, is_terminate = self.step(action, node_path, edge_path)

                # Learn from the outcome by updating Q-table
                self.learn(last_state, action, next_state, reward)

                # Update state
                if last_state != next_state:  # last_state == next_state only if the action is not valid
                    edge_path.append(next_edge)
                    node_path.append(next_state)

            # Append to logs
            self.logs[episode] = [node_path, edge_path]

            # Deal with convergence: > threshold to make same results for needed times, and make sure reach the end node
            if episode > threshold and self.logs[episode][0][-1] == self.env.end_node:

                # Convergence when 5 same routes produced consecutively
                threshold_lst = list(self.logs.values())[-threshold:]
                if all(x == threshold_lst[0] for x in threshold_lst):
                    end_time = datetime.datetime.now()  # record ending time
                    time_difference = end_time - start_time
                    processing_seconds = time_difference.total_seconds()

                    # --- results output ---
                    print('Training Completed...\n')
                    print(f'-- Last Episode: {episode}')
                    print(f'-- States: {self.logs[episode][0]}')
                    print(f'-- Edges: {self.logs[episode][1]}')
                    print(f'-- Processing Time: {processing_seconds} seconds')

                    if self.env.evaluation in ("time"):
                        print(f'-- Travelled Time: {round((self.env.get_edge_time(self.logs[episode][1])+  self.env.get_tl_offset(self.logs[episode][1]))/60, 2)} mins')
                    else:
                        print(f'-- Travelled Distance: {round(self.env.get_edge_distance(self.logs[episode][1]), 2)} m')

                    return self.logs[episode][0], self.logs[episode][1], episode, self.logs

            # Deal with the case that it is unable to converge
            if episode+1 == num_episodes:
                print('Training Completed...\n')
                end_time = datetime.datetime.now()
                time_difference = end_time - start_time
                processing_seconds = time_difference.total_seconds()
                print(f'-- Processing Time: {processing_seconds} seconds')
                sys.exit(f'Cannot find shortest route within {num_episodes} episodes')


class Q_Learning(rl_agent):
    def __init__ (self, env, start_node, end_node, learning_rate = 0.9, discount_factor = 0.1):
        super().__init__(env, start_node, end_node, learning_rate, discount_factor)  # inherit from parent class


    def act(self, state):
        # Choose action with highest Q-value
        state_index = self.env.state_space.index(state)
        action = np.argmax(self.q_table[state_index])
        return action


class SARSA(rl_agent):
    def __init__ (self, env, start_node, end_node, learning_rate = 0.9, discount_factor = 0.1, exploration_rate = 0.1):
        super().__init__(env, start_node, end_node, learning_rate, discount_factor)  # inherit from parent class

        self.exploration_rate = exploration_rate  # Define additional parameter


    def act(self, state):
        if np.random.random() < self.exploration_rate:
            # Exploration
            action = np.random.choice(len(self.env.action_space))
        else:
            # Exploitation
            state_index = self.env.state_space.index(state)
            action = np.argmax(self.q_table[state_index])
        return action
