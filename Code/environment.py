import numpy as np


class Environment:
    def __init__(self, num_of_nodes, values_per_node, neighbours_matrix, data_node = 3, start_node = 0, detection_probability = 1, initial_security = 2):
        
        self.neighbours_matrix = neighbours_matrix
        self.detection_probability = detection_probability
        self.start_node = start_node
        self.data_node = data_node
        self.num_of_nodes = num_of_nodes
        self.values_per_node = values_per_node
        self.attacker_detected = False
        self.data_node_cracked = False
        self.attacker_attack_successful = False

        self.attacker_nodes_with_maximal_value = []
        self.defender_nodes_with_maximal_value = []
        self.defender_nodes_with_maximal_value_dqn = []
        self.attacker_maxed_all_values = False
        self.defender_maxed_all_values = False

        self.rng = np.random.default_rng(42)
        
        self.initial_security = initial_security
        self.attack_values = np.zeros((self.num_of_nodes,self.values_per_node))
        self.defence_values = np.zeros((self.num_of_nodes,self.values_per_node + 1))+initial_security # +1 because there is the detection probability
        

        for i in range(1,num_of_nodes):
            rnd = self.rng.integers(values_per_node)
            self.defence_values[i][rnd] = 0
            self.defence_values[i][-1] = self.detection_probability # the last value is the detection probability

        self.defence_values[start_node]-=self.initial_security 
        self.defence_values[start_node][-1] = 0
    
        
        
       
       
    def reset(self):
        self.attacker_detected = False
        self.data_node_cracked = False
        self.attacker_attack_successful = False

        self.attacker_nodes_with_maximal_value = []
        self.defender_nodes_with_maximal_value = []
        self.defender_nodes_with_maximal_value_dqn = []
        self.attacker_maxed_all_values = False
        self.defender_maxed_all_values = False

        self.attack_values = np.zeros((self.num_of_nodes,self.values_per_node))
        self.defence_values = np.zeros((self.num_of_nodes,self.values_per_node + 1))+self.initial_security

        self.rng = np.random.default_rng(42)

        for i in range(1,self.num_of_nodes):
            rnd = self.rng.integers(self.values_per_node)
            self.defence_values[i][rnd] = 0
            self.defence_values[i][-1] = self.detection_probability

        self.defence_values[self.start_node]-= self.initial_security 
        self.defence_values[self.start_node][-1] = 0

    def do_defender_action(self,node,defence_type):
        self.defence_values[node][defence_type] += 1
        if self.defence_values[node][defence_type] == 10:
            self.defender_nodes_with_maximal_value.append((node, defence_type))
        if len(self.defender_nodes_with_maximal_value) == self.num_of_nodes*(self.values_per_node+1):
            self.defender_maxed_all_values = True
        return

    def do_attacker_action(self,node,attack_type):
        self.attacker_attack_successful = False
        self.attack_values[node][attack_type] += 1
        if self.attack_values[node][attack_type] == 10:
            self.attacker_nodes_with_maximal_value.append((node, attack_type))

        if len(self.attacker_nodes_with_maximal_value) == self.num_of_nodes*self.values_per_node:
            self.attacker_maxed_all_values = True
        
        if self.attack_values[node][attack_type] > self.defence_values[node][attack_type]: #successful attack
            if node == self.data_node:
                self.data_node_cracked = True
                #print("NETWORK CRAKED DATA NODE")
                return (100,-100)
            else:
                #print("NETWORK ATTACKER SUCCESSFUL ATTACK")
                self.attacker_attack_successful = True
                return (0,0)
        else:
            #print("NETWORK ATTACKER BLOCKED ATTACK")
            if np.random.default_rng().random()*10 < self.defence_values[node][-1]: #detected?
                self.attacker_detected = True
                #print("NETWORK ATTACKER DETECTED")
                return (-100,100)
            return (0,0)
    

    def termination_condition(self):
        if self.data_node_cracked or self.attacker_detected:
            return True
        if self.attacker_maxed_all_values or self.defender_maxed_all_values:
            return True
        return False
    

    
