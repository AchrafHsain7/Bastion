from library import *


# FINALLY REMOVE THE VPN BYPASS


# 1 means attack path exists from col idx host to row idx host
topology = [[1, 0, 0, 0, 0], 
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1]
            ]  


host_states = ["C", "X", "D", "I", "Z"]
# GW, Web, WS, DB, BK
state0 = ["X", "C", "C", "C", "C", "A"]
# MDP actions: (action type, host)
actions_type_D = {"Noop":0, "Monitor":1, "Isolate":2, "Restore":3, "Fix":4}
actions_type_A = {"Noop":0, "Spread":1, "Destroy":2}
mdp_transitions_D = {   "C":["C", "C","I","C", "C"],
                        "X":["X", "D", "I", "X", "X"],   
                        "D" : ["D", "D", "I", "D", "D"],
                        "I":["I", "I", "I", "C", "I"],
                        "Z":["Z", "Z", "Z", "Z", "C"]
            }
mdp_transitions_A = {   "C" : ["C", "X", "C"], #Transition function must use table + check adjacency conditions
                        "X" : ["X", "X", "Z"],
                        "D" : ["D", "D", "Z"],
                        "I": ["I", "I", "I"],
                        "Z" : ["Z", "Z", "Z"]
                }

# Defense Spec: 
# 1 - always not both DB and BK in Destroy/Compromise??
# 2 - at least 3 hosts working at all times (not isol or destroyed)
qD_states = [["q0", "q1", "viol"], ["safe", "viol"]]
delta1 = {"q0":{"0":"q0", "1":"q1", "2":"viol"}, "q1":{"0":"q0", "1":"q1", "2":"viol"} ,"viol":{"0":"viol", "1":"viol", "2":"viol"}}
automaton_D1 = SafetyAutomaton("q0", delta1)
qD_states2 = ["safe", "viol"]
delta1_2 = {"safe":{"1":"safe", "0":"viol"}, "viol":{"1":"viol", "0":"viol"},}
automaton_D2 = SafetyAutomaton("safe", delta1_2)

automaton_D = ProductAutomaton([automaton_D1, automaton_D2])

# Attacker Spec: At most two destroy
qA_states = [["q0", "q1", "q2", "viol"]]
delta2 = {"q0":{"other":"q0", "destroy":"q1"}, "q1":{"other":"q1", "destroy":"q2"}, "q2":{"other":"q2", "destroy":"viol"}, "viol":{"other":"viol", "destroy":"viol"}}
automaton_A = SafetyAutomaton("q0", delta2)


class Labeler(LabelingFunction):
        # mdp state: (X, I), Action: (attk_idx, host)
        def labeler_D(self, mdp_state, action, debug=False):
            symbols = []
            # First condition: Bk and DB never both compromised/Destroyed/Detected
            if mdp_state[-3] in ["Z", "X", "D"] and  mdp_state[-2] in ["Z", "X", "D"]:
                s = "2"
            elif mdp_state[-3] in ["Z", "X", "D"] or  mdp_state[-2] in ["Z", "X", "D"]:
                s = "1"
            else:
                s = "0"
            symbols.append(s)

            # second confition: At least 3 hosts not in Isol/Destroyed
            working_hosts = sum(h not in ["Z", "I"] for h in mdp_state[:-1])
            # if debug: 
            #     print(mdp_state)
            #     print(working_hosts)
            if working_hosts >= 3:
                s = "1"
            else: 
                s = "0"
            symbols.append(s)

            return symbols
        
        def labeler_A(self, mdp_state, action):
            # Attacker state never change based on defender given current specs
            if mdp_state[-1] == "A": return None #next state is given meaning last last action was done by defender
            if action[0] == 2: # 2 is index of Destroy
                return ["destroy"]
            return ["other"]