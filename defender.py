import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import itertools

class Defender:
    def __init__(self, environment, learning = True, learning_algorithm = "montecarlo", exploration = "epsilon_greedy", algorithms_parameters = None):

        if algorithms_parameters == None:
            self.algorithms_parameters = {  
                                        ("montecarlo", "epsilon_greedy"): {
                                            "alpha" : 0.05,
                                            "epsilon" : 0.1
                                        },
                                        ("montecarlo", "softmax"): {
                                            "alpha" : 0.1,
                                            "tau" : 4
                                        },
                                        ("qlearning", "epsilon_greedy"): {
                                            "alpha" : 0.05,
                                            "gamma" : 0.91,
                                            "epsilon" : 0.07
                                        },
                                        ("dqn", "epsilon_greedy"): {
                                            "eta" : 0.001,
                                            "gamma" : 0.95,
                                            "epsilon" : 0.05,
                                            "batch_size" : 15
                                        }
                                    }
        else:
            self.algorithms_parameters = algorithms_parameters

        self.model = None
        self.replay_buffer = None
        self.optimizer = None
        self.loss_fn = None
        self.Q_table = None
        self.returns = None


        self.environment = environment
        self.learning = learning
        self.learning_algorithm = learning_algorithm
        self.exploration = exploration
        self.rng = np.random.default_rng()

        if self.learning_algorithm == "dqn":
            self.model = keras.Sequential([
                keras.layers.Flatten(input_shape=(self.environment.num_of_nodes,self.environment.values_per_node+1)),
                keras.layers.Dense(6, activation="sigmoid"),
                keras.layers.Dense((self.environment.values_per_node+1)*self.environment.num_of_nodes)
            ])
            self.replay_buffer = deque(maxlen=2000)
            self.optimizer = keras.optimizers.Adam(lr=self.algorithms_parameters[("dqn","epsilon_greedy")]["eta"])
            self.loss_fn = keras.losses.mean_squared_error
        else:
            self.Q_table = np.zeros((environment.num_of_nodes, environment.values_per_node + 1))
            self.returns = {}

        return

    def reset(self):
        return


    def select_random_action(self, environment): 
        actions = itertools.product(range(environment.num_of_nodes),range(environment.values_per_node+1))
        actions = [x for x in actions if x not in environment.defender_nodes_with_maximal_value]
        node, defence_type = self.rng.choice(actions)
        return (node,defence_type)

    def dqn_epsilon_greedy(self, environment, epsilon):
        x = self.rng.random()
        if x < epsilon:
            return self.select_random_action(environment)
        else:
            Q_values = self.model.predict(environment.defence_values[np.newaxis])
            actions = [x for x in np.argsort(Q_values, axis=None) if x not in environment.defender_nodes_with_maximal_value_dqn]
            node, defence_type = divmod(actions[1], environment.values_per_node + 1)
            return (node, defence_type)

    def softmax(self,x):
        return np.exp(x)/sum(np.exp(x))

    def select_action_softmax(self, environment, tau):
        actions = {}
        for node in range(environment.num_of_nodes):
            for defence_type in range(environment.values_per_node + 1):
                if (node,defence_type) not in environment.defender_nodes_with_maximal_value:
                    actions[(node, defence_type)] = self.Q_table[node][defence_type]/tau

        probabilities = self.softmax(np.array(list(actions.values())))
        node, defence_type = self.rng.choice(np.array(list(actions.keys())),p=probabilities)

        return (node, defence_type)


    def select_action_epsilon_greedy(self, environment, e):
        x = self.rng.random()
        if x < e:
            return self.select_random_action(environment)
        else:
            actions = np.unravel_index(np.argsort(self.Q_table, axis=None), self.Q_table.shape)
            actions = [(x,y) for (x,y) in zip(actions[0],actions[1]) if (x,y) not in environment.defender_nodes_with_maximal_value]
            node, defence_type = actions[-1]

            return (node,defence_type)

    def select_action(self, environment, episode):
        if self.learning:
            if self.learning_algorithm == "dqn":
                return self.dqn_epsilon_greedy(environment, self.algorithms_parameters[(self.learning_algorithm, self.exploration)]["epsilon"])
            else:
                if episode < 5:
                    return self.select_random_action(environment) 
                elif self.exploration == "epsilon_greedy":
                    return self.select_action_epsilon_greedy(environment, self.algorithms_parameters[(self.learning_algorithm, self.exploration)]["epsilon"])
                elif self.exploration == "softmax":
                    return self.select_action_softmax(environment, self.algorithms_parameters[(self.learning_algorithm, self.exploration)]["tau"])
        else:
            return self.select_random_action(environment)

    def MonteCarloUpdate(self, sas, alpha, R):
        sas.reverse()
        for i in range(len(sas)):
            node, defence_type = sas[i][1]
            self.Q_table[node][defence_type] += alpha*(R - self.Q_table[node][defence_type])
        return


    def MonteCarloUpdateTraditional(self,sars, gamma):

        sars.reverse()
        G = 0

        for i in range(len(sars)):
            s,a,r = sars[i]
            G = gamma*G + r
            if not (s,a) in [(x[0], x[1]) for x in sars[i+1:]]: # first visit Monte Carlo
                if self.returns.get((s,a)):
                    self.returns[(s,a)].append(G)
                else:
                    self.returns[(s,a)] = [G]
                
                self.Q_table[a[0]][a[1]] = sum(self.returns[(s,a)]) / len(self.returns[(s,a)]) 

        return

    def QLearningUpdate(self, sas, alpha, gamma, R):
        sas.reverse()
        best_q_value_next_state = 0
        for i in range(len(sas)):
            node, defence_type = sas[i][1]
            if i == 0: # last state
                self.Q_table[node][defence_type] += alpha*(R - self.Q_table[node][defence_type])
            else:
                self.Q_table[node][defence_type] += alpha*(gamma*best_q_value_next_state - self.Q_table[node][defence_type])
            
            best_q_value_next_state = self.Q_table.max()
        return

    def QValuesUpdate(self, sas, R):
        if self.learning_algorithm == "montecarlo":
            self.MonteCarloUpdate(sas, self.algorithms_parameters[(self.learning_algorithm, self.exploration)]["alpha"], R)
        elif self.learning_algorithm == "qlearning":
            self.QLearningUpdate(sas, self.algorithms_parameters[(self.learning_algorithm, self.exploration)]["alpha"], 
                                self.algorithms_parameters[(self.learning_algorithm, self.exploration)]["gamma"], R)



    def to_flatten_index(self, action):
        return action[0]*(self.environment.values_per_node + 1) + action[1]

    # Code made by Mirco Musolesi (2020)
    # https://www.mircomusolesi.org/courses/AAS19-20/AAS19-20_CartPole-DQN.ipynb

    def append_experience(self, experience):
        self.replay_buffer.append(experience)
        return

    def sample_experiences(self, batch_size):
        indices = self.rng.integers(len(self.replay_buffer), size=batch_size)
        batch = [self.replay_buffer[index] for index in indices]
        states, actions, rewards, next_states, dones = [np.array([experience[field_index] for experience in batch]) for field_index in range(5)]
        return states, actions, rewards, next_states, dones
    
    def dqn_training_step(self):
        experiences = self.sample_experiences(self.algorithms_parameters[("dqn","epsilon_greedy")]["batch_size"])
        states, actions, rewards, next_states, dones = experiences
        actions = [self.to_flatten_index(action) for action in actions]
        next_Q_values = self.model.predict(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + (1-dones)*self.algorithms_parameters[("dqn","epsilon_greedy")]["gamma"]*max_next_Q_values)
        mask = tf.one_hot(actions, (self.environment.values_per_node +1 ) * self.environment.num_of_nodes)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values*mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values,Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return