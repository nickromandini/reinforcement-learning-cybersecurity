import numpy as np

from attacker import Attacker
from defender import Defender
from environment import Environment
import matplotlib.pyplot as plt
import itertools




def episode(environment,attacker,defender,episode_number):

    #print("{} episode".format(episode_number))

    environment.reset()
    attacker.reset()
    defender.reset()

    attacker_sas = []
    defender_sas = []

    
    iterations = 0
    while True:
        #print("{} iteration".format(iterations))
        #print("attack values")
        #print(environment.attack_values)
        #print("defence values")
        #print(environment.defence_values)
        #print("defender select action")
        defender_action = defender.select_action(environment, episode_number)
        #if defender.learning_algorithm == "dqn":
        #    print("DEFENDER ACTION: action {}".format(defender_action))
        #else:
        #    print("DEFENDER ACTION: node {}, defence_type {}".format(defender_action[0],defender_action[1]))
        #print("attacker select action")
        attacker_action = attacker.select_action(environment, episode_number)
        #print("ATTACKER ACTION: node {}, attack_type {}".format(attacker_action[0],attacker_action[1]))
        environment.do_defender_action(*defender_action)
        if defender.learning and defender.learning_algorithm == "dqn":
            defender_current_state = environment.defence_values

        attacker_current_reward, defender_current_reward = environment.do_attacker_action(*attacker_action)

        #attacker_rewards.append(attacker_current_reward)
        #defender_rewards.append(defender_current_reward)

        attacker_sas.append((attacker.current_node, attacker_action)) #, attacker_current_reward))
        defender_sas.append((0,defender_action)) #,defender_current_reward))

        if environment.attacker_attack_successful: #successful attack
            attacker.update_position(attacker_action[0])

        if environment.termination_condition():
            if defender.learning and defender.learning_algorithm == "dqn":
                defender.append_experience((defender_current_state, defender_action, defender_current_reward/100, environment.defence_values, 1))
            break

        if defender.learning and defender.learning_algorithm == "dqn":
            defender.append_experience((defender_current_state, defender_action, defender_current_reward, environment.defence_values , 0))
        
        

        iterations+=1
    
    #print("Episode ended")
    attacker_final_reward = 0
    defender_final_reward = 0
    if environment.attacker_maxed_all_values or environment.defender_maxed_all_values:
        return (0,0)
    if environment.data_node_cracked:
        #print("Attacker wins!")
        attacker_final_reward = 100
        defender_final_reward = -100
    elif environment.attacker_detected:
        #print("Defender wins!")
        attacker_final_reward = -100
        defender_final_reward = 100

    if attacker.learning:
        attacker.QValuesUpdate(attacker_sas, attacker_final_reward)

    if defender.learning:
        if defender.learning_algorithm == "dqn" and episode_number > 20:
            defender.dqn_training_step()
        else:
            defender.QValuesUpdate(defender_sas, defender_final_reward)


    return attacker_final_reward, defender_final_reward
    

            
if __name__ == '__main__':

    neighbours_matrix = np.array([[1,2],
                           [0,3],
                           [0,3],
                           [1,2]])
    num_of_nodes = 4
    data_node = 3

    simulations = 10
    
    episodes = 20000
    simulation_attacker_combinations = [ [False, None, None], [True, "montecarlo", "epsilon_greedy"], [True, "montecarlo", "softmax"], [True, "qlearning", "epsilon_greedy"] ]
    simulation_defender_combinations = [ [False, None, None], [True, "montecarlo", "epsilon_greedy"], [True, "montecarlo", "softmax"], [True, "qlearning", "epsilon_greedy"], [True, "dqn", "epsilon_greedy"] ]
    
    simulation_combinations = itertools.product(simulation_attacker_combinations,simulation_defender_combinations)

    for combination in simulation_combinations:
        
        attacker_win_percentages = []
        defender_win_percentages = []

        print("Attacker combination: {}".format(combination[0]))
        print("Defender combination: {}".format(combination[1]))
        
        for sim in range(simulations):
            #print("Simulation {}".format(sim))
            environment = Environment(num_of_nodes=num_of_nodes,values_per_node=10, neighbours_matrix = neighbours_matrix, detection_probability= 1, data_node=data_node)
            attacker = Attacker(environment, learning = combination[0][0], learning_algorithm = combination[0][1], exploration= combination[0][2])
            defender = Defender(environment, learning = combination[1][0], learning_algorithm = combination[1][1], exploration= combination[1][2])
           


            attacker_total_rewards = []
            defender_total_rewards = []

            attacker_wins = 0
            defender_wins = 0
            ep = 0

            while ep < episodes:

                r_a,r_d = episode(environment,attacker, defender, ep)

                if r_a == 0: #tie
                    continue

                attacker_total_rewards.append(r_a)
                defender_total_rewards.append(r_d)
                if r_a > r_d:
                    attacker_wins+=1
                else:
                    defender_wins+=1
                
                ep += 1
            
            
            #print("ATTACKER Q_TABLE")
            #print(attacker.Q_table)
            #print("DEFENDER Q_TABLE")
            #print(defender.Q_table)

            attacker_win_percentages.append(attacker_wins/episodes)
            defender_win_percentages.append(defender_wins/episodes)


        

        mean_attacker_win_percentage = np.mean(np.array(attacker_win_percentages))
        mean_defender_win_percentage = np.mean(np.array(defender_win_percentages))
        
        print("Attacker mean win percentage {}".format(mean_attacker_win_percentage))
        print("Defender mean win percentage {}".format(mean_defender_win_percentage))


