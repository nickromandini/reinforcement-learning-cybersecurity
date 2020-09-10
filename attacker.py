import numpy as np
import os
import itertools

class Attacker:

    

    def __init__(self, environment, learning = True, learning_algorithm = "montecarlo", exploration= "epsilon_greedy", algorithms_parameters = None):

        if algorithms_parameters == None:
            self.algorithms_parameters = {  
                                        ("montecarlo", "epsilon_greedy"): {
                                            "alpha" : 0.05,
                                            "epsilon" : 0.05
                                        },
                                        ("montecarlo", "softmax"): {
                                            "alpha" : 0.05,
                                            "tau" : 5
                                        },
                                        ("qlearning", "epsilon_greedy"): {
                                            "alpha" : 0.1,
                                            "gamma" : 0.95,
                                            "epsilon" : 0.04
                                        }
                                    }
        else:
            self.algorithms_parameters = algorithms_parameters

        self.current_node = environment.start_node
        self.environment = environment
        self.learning = learning
        self.learning_algorithm = learning_algorithm
        self.exploration = exploration
        self.rng = np.random.default_rng()
        self.returns = {}
        self.Q_table = np.zeros((environment.num_of_nodes-1, environment.num_of_nodes, environment.values_per_node))

        return

    def reset(self):
        self.current_node = self.environment.start_node


    def select_random_action(self, environment):
        neighbours = environment.neighbours_matrix[self.current_node] 
        actions = itertools.product(neighbours,range(environment.values_per_node))
        actions = [x for x in actions if x not in environment.attacker_nodes_with_maximal_value]
        node, attack_type = self.rng.choice(actions)
        return (node,attack_type)

    def softmax(self,x):
        return np.exp(x)/sum(np.exp(x))

    def select_action_softmax(self, environment, tau):
        neighbours = environment.neighbours_matrix[self.current_node]
        actions = {}
        for neighbour in neighbours:
            for attack_type in range(environment.values_per_node):
                if (neighbour, attack_type) not in environment.attacker_nodes_with_maximal_value:
                    actions[(neighbour, attack_type)] = self.Q_table[self.current_node][neighbour][attack_type]/tau

        probabilities = self.softmax(np.array(list(actions.values())))
        node, attack_type = self.rng.choice(np.array(list(actions.keys())),p=probabilities)

        return (node, attack_type)

    def select_action_epsilon_greedy(self, environment, e):
        neighbours = environment.neighbours_matrix[self.current_node]
        x = self.rng.random()
        if x < e:
            return self.select_random_action(environment)
        else:
            actions = np.unravel_index(np.argsort(self.Q_table[self.current_node], axis=None), self.Q_table[self.current_node].shape)
            actions = [(x,y) for (x,y) in zip(actions[0],actions[1]) if (x,y) not in environment.attacker_nodes_with_maximal_value and x in neighbours]
            node, attack_type = actions[-1]

            return (node,attack_type)

    def select_action(self, environment, episode):
        if self.learning and episode > 5:
            if self.exploration == "epsilon_greedy":
                return self.select_action_epsilon_greedy(environment, self.algorithms_parameters[(self.learning_algorithm, self.exploration)]["epsilon"])
            elif self.exploration == "softmax":
                return self.select_action_softmax(environment, self.algorithms_parameters[(self.learning_algorithm, self.exploration)]["tau"])
        else:
            return self.select_random_action(environment)

    def update_position(self,node):
        self.current_node = node


    


    def MonteCarloUpdate(self, sas, alpha, R):
        sas.reverse()
        for i in range(len(sas)):
            s = sas[i][0]
            node, attack_type = sas[i][1]
            self.Q_table[s][node][attack_type] += alpha*(R - self.Q_table[s][node][attack_type])
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

                self.Q_table[s][a[0]][a[1]] = sum(self.returns[(s,a)]) / len(self.returns[(s,a)]) 
                '''
                if self.Q_table.get(s):
                    if self.Q_table[s].get(a):
                        self.Q_table[s][a] = sum(self.returns[(s,a)]) / len(self.returns[(s,a)]) 
                    else:
                        self.Q_table[s].update({a: sum(self.returns[(s,a)]) / len(self.returns[(s,a)])})
                else:
                    self.Q_table[s] = {a: sum(self.returns[(s,a)]) / len(self.returns[(s,a)])}
                '''

        return
        
    def QLearningUpdate(self, sas, alpha, gamma, R):
        sas.reverse()
        best_q_value_next_state = 0

        for i in range(len(sas)):
            s = sas[i][0]
            node, attack_type = sas[i][1]

            if i == 0: # last state
                self.Q_table[s][node][attack_type] += alpha*(R - self.Q_table[s][node][attack_type])
            else:
                self.Q_table[s][node][attack_type] += alpha*(gamma*best_q_value_next_state - self.Q_table[s][node][attack_type])
            
            best_q_value_next_state = self.Q_table[s].max()
        return


    def QValuesUpdate(self, sas, R):
        if self.learning_algorithm == "montecarlo":
            self.MonteCarloUpdate(sas, self.algorithms_parameters[(self.learning_algorithm, self.exploration)]["alpha"], R)
        elif self.learning_algorithm == "qlearning":
            self.QLearningUpdate(sas, self.algorithms_parameters[(self.learning_algorithm, self.exploration)]["alpha"], 
                                self.algorithms_parameters[(self.learning_algorithm, self.exploration)]["gamma"], R)