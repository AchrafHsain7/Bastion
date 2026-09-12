from abc import ABC, abstractmethod
from collections import defaultdict
import itertools
from logging import debug
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 




# Labeling Function ==========================================================================================
class LabelingFunction(ABC):
    # mdp state: (h1, ..., hn, Player), Action: (action_idx, host)
    @abstractmethod
    def labeler_D(self, mdp_state, action)->list|None:
        pass
    @abstractmethod
    def labeler_A(self, mdp_state, action)->list|None:
        pass


# Automatons ======================================================================================================
class SafetyAutomaton:
    def __init__(self, q0, transitions) -> None:
        self.q0 = q0
        self.transitions = transitions 
        self.state = q0

    def step(self, symbols):
        if symbols == None:
            return
        if type(symbols)==list: symbols=symbols[0]
        self.state = self.transitions[self.state][symbols]

    def frozen_step(self, state, symbols):
        if symbols == None:
            return
        if type(symbols)==list: symbols=symbols[0]
        return self.transitions[state][symbols]
    
    def reset(self):
        self.state = self.q0


class ProductAutomaton:
    def __init__(self, automatons) -> None:
        self.automatons = automatons
        self.num_automatons = len(automatons)
        self.state = tuple([a.state for a in automatons])
    
    def step(self, symbols):
        #symbols is a list one for each automaton
        # If no symbol passed then do nothing
        if symbols == None:
            return
        for i in range(self.num_automatons):
            self.automatons[i].step(symbols[i])
        self.state = tuple([a.state for a in self.automatons])
    
    def frozen_step(self, states, symbols):
        # list of states and list of symbols
        # If no symbol provided by labeling function then no automaton state change
        if symbols == None:
            return tuple(states)
        dfa_state = []
        for i in range(self.num_automatons):
            dfa_state.append(self.automatons[i].frozen_step(states[i], symbols[i]))
        return tuple(dfa_state)
    
    def reset(self):
        for i in range(self.num_automatons):
            self.automatons[i].reset()
        self.state = tuple([a.state for a in self.automatons])


# Environment ========================================================================================================
class MDP:
    def __init__(self, topology, state0, mdp_transitions_D, mdp_transitions_A) -> None:
        self.state0 = state0
        self.topology = topology
        self.mdp_transitions_A = mdp_transitions_A
        self.mdp_transitions_D = mdp_transitions_D
    
    def next_state(self, state, action):
        # mdp state in form (h1, h2, ...., hn, Player) | action in form (type, host) ex: (2: "Destroy", 1)
        state = list(state)
        action_idx, host = action
        host_state = state[host]
        if state[-1] == "D":
            state[host] = self.mdp_transitions_D[host_state][action_idx]
            state[-1] = "A"
            return state

        elif state[-1] == "A":
            if action_idx == 1: # check if a neighbor is compromised
                spreadable = False
                for h in range(len(self.topology[host])):
                    if h!=host and self.topology[host][h] == 1 and state[h] in ["X", "D"]:
                        spreadable = True
                        break
                if not spreadable: 
                    #no change
                    state[-1] = "D"
                    return state
            
            state[host] = self.mdp_transitions_A[host_state][action_idx]
            state[-1] = "D"
            return state

    def reward_function(self, old_state, state):
        # reward function
        num_hosts = len(state[:-1])
        num_clean = sum(1 for h in state[:-1] if h=="C")
        num_compromised = num_hosts - num_clean
        # Tag of war: Reward between -1 and 1, for defendder having >50% clean give positive reward, opposite for attacker <50% gives positive reward
        reward = (num_clean - num_compromised)/num_hosts
            
        # new state is attacker state so old action taken by defender
        if old_state[-1] == "D": return reward
        #new state is defender state so old action taken by attacker
        if old_state[-1] == "A": return -reward
            
        

    # RL training env step
    def step(self, state, action):
        state = list(state)
        old_state = state.copy()
        action_idx, host = action
        host_state = state[host]
        if state[-1] == "D":
            state[host] = self.mdp_transitions_D[host_state][action_idx]
            state[-1] = "A"
            return state, self.reward_function(old_state, state)

        elif state[-1] == "A":
            if action_idx == 1: # check if neighbor compromised
                spreadable = False
                for h in range(len(self.topology[host])):
                    if h!=host and self.topology[host][h] == 1 and state[h] in ["X", "D"]:
                        spreadable = True
                        break
                if not spreadable: 
                    #no change
                    state[-1] = "D"
                    return state, self.reward_function(old_state, state)
            
            state[host] = self.mdp_transitions_A[host_state][action_idx]
            state[-1] = "D"
            return state, self.reward_function(old_state, state)
        



class NetworkAnalyzer:
    def __init__(self, topology, host_states, state0, mdp_transitions_D, mdp_transitions_A,  actionsD, actionsA, automaton_D, automaton_A , qD_states, qA_states, labeler) -> None:
        self.topology = topology
        self.host_states = host_states
        self.num_hosts = len(topology)

        self.automatonD = automaton_D
        self.automatonA = automaton_A
        self.qD_states = qD_states #list of list of states, a list of states per automaton/spec
        self.qA_states = qA_states

        
        self.mdp = MDP(topology, state0, mdp_transitions_D, mdp_transitions_A)
        self.actionsD = actionsD
        self.actionsA = actionsA
        self.max_actionD = self.num_hosts*len(actionsD) 
        self.max_actionA = self.num_hosts*len(actionsA) 
        self.mdp_product = self.compute_mdp_product()
        self.map_idx = {self.mdp_product[i]:i for i in range(len(self.mdp_product))}
        self.in_attractor = [0]*len(self.mdp_product)

        self.labeler = labeler

        self.successors = self.create_successor_graph()


    def compute_mdp_product(self):
        print("CREATING PRODUCT MDP")
        #qD_state and qA_states are expected to be list of lists ex: [[safe, viol]] or [[q1, q2, viol], [safe, viol]]
        mdp_product = []
        if len(self.qD_states) > 1:
            qD_product = list(itertools.product(*self.qD_states))
        else:
            qD_product = self.qD_states[0]
        if len(self.qA_states) > 1:
            qA_product = list(itertools.product(*self.qA_states))
        else:
            qA_product = self.qA_states[0]
        mdp_product = list(itertools.product(*([self.host_states]*self.num_hosts), ["A","D"], qD_product, qA_product))

        return mdp_product

    def create_successor_graph(self):
        # Successor Graph
        #indexes: -1:qA, -2:qD, -3:Player
        print("CREATING SUCCESSOR GRAPH")
        successors = {}
        for s in tqdm(self.mdp_product):
            s = tuple(s)
            actions = len(self.actionsD) if s[-3]=="D" else len(self.actionsA)
            actions = itertools.product(range(actions), range(self.num_hosts))
            successors[s] = [] 
            for a in actions:
                next_mdp = self.mdp.next_state(s[:-2], a)
                # Labeler class should check if the automaton need to move or not, return None if not, should accpet both mdp_state and action (,)
                next_mdp.append(self.automatonD.frozen_step(s[-2], self.labeler.labeler_D(next_mdp, a))) # Defender automaton may move since depend on state which changed
                if s[-3] == "A":
                    next_mdp.append(self.automatonA.frozen_step(s[-1], self.labeler.labeler_A(next_mdp, a))) #
                else:
                    next_mdp.append(s[-1])
                successors[s].append(tuple(next_mdp))

            # memory performance in case needed but getting safe actions become harder
            # unique = list({tuple(x) for x in successors[s]})
            # successors[s] = unique
            
        return successors

    
    def compute_attractor(self):
        attractor_sizes = []
        print("COMPUTING ATTRACTOR")
        attractor0 = []
        
        for i, s in enumerate(self.mdp_product):

            if "viol" in s[-2]: #defense/attacker violation
                attractor0.append(tuple(s))
                self.in_attractor[i] = 1

        print("Attr0", len(attractor0))
        attractor_sizes.append(len(attractor0))
        attractor_old = attractor0
        in_attractor_old = self.in_attractor.copy()
        idx = 1
        while True:
            attractor = attractor_old.copy()
            in_attractor_old = self.in_attractor.copy()
            for i, s in enumerate(tqdm(self.mdp_product)):
                if in_attractor_old[i]==1: continue
                s = tuple(s)
                if s[-3] == "A":
                    add = False
                    # if one next successor LEGAL vertext is in attractor then add this one as well
                    for j, next_s in enumerate(self.successors[s]):
                        if in_attractor_old[self.map_idx[next_s]]==1 and next_s[-1]!="viol":
                            add = True
                            break
                    if add and self.in_attractor[i]==0:
                        attractor.append(s)
                        self.in_attractor[i] = 1
                    
                elif s[-3] == "D":
                    # if all next states in attractor then add this state to attractor
                    add = True
                    for next_s in self.successors[s]:
                        if in_attractor_old[self.map_idx[next_s]]==0: # not in attractor
                            add = False
                            break
                    if add and self.in_attractor[i]==0:
                        attractor.append(s)
                        self.in_attractor[i] = 1
            
            
            print(f"===> Attr{idx}=", len(attractor), end="\n\n")
            attractor_sizes.append(len(attractor))
            if len(attractor) == len(attractor_old):
                self.attractor = attractor
                self.attractor_sizes = attractor_sizes[:-1]
                return attractor, self.attractor_sizes
            
            attractor_old = attractor
            idx+=1

    def compute_winning_region(self):
        if sum(self.in_attractor)==0:
            self.attractor, _ = self.compute_attractor()
        print("COMPUTING WINNING REGION")
        winning_region = []
        for i, s in enumerate(tqdm(self.mdp_product)):
            s = tuple(s)
            if self.in_attractor[i]==0:
                winning_region.append(s)
        self.winning_region = winning_region
        return winning_region

    
    def action2idx(self, action):
        # action = (type, host)
        idx = (self.num_hosts)*action[0] + action[1]
        return idx
    
    def idx2action(self, idx):
        action = (idx//(self.num_hosts), idx%(self.num_hosts))
        return action

    def compute_safe_actions(self):
        print("COMPUTING DEFENDER SAFE ACTIONS")
        safe_actions = {}
        for s in tqdm(self.successors):
            safe_actions[s] = []
            for i, next_s in enumerate(self.successors[s]):
                if s[-3] == "A" and next_s[-1]!="viol":
                    if self.in_attractor[self.map_idx[next_s]]==0: safe_actions[s].append(i)
                elif s[-3] == "D":
                    if self.in_attractor[self.map_idx[next_s]]==0: safe_actions[s].append(i)
        self.safe_actions = safe_actions

        # print(self.safe_actions.keys())
        if len(self.safe_actions[("X", "C", "C", "C", "C", "A", ("q0", "safe"), "q0")]) == 0: 
            print("Not defensible")
        else:
            print("Defensible")
        return safe_actions

    
    # Metrics
    def attackability(self, discount=None):
        attack = 0
        for i in range(1, len(self.attractor_sizes)):
            shell_size = self.attractor_sizes[i] - self.attractor_sizes[i-1]
            if discount == None: weight = 1/i #linear decay
            else: weight = discount**i  #exponential decay that reflect temporal horizon of agent
            attack += weight*shell_size 
        #normalize by reachable state space/ we will just use winning region for now
        attack /= len(self.winning_region)
        return attack

    # Sinked states: states that sinked during the attractor computation into violation region, not initially violations
    def sinked_states(self):
        return  (self.attractor_sizes[-1] - self.attractor_sizes[0])/(len(self.mdp_product) - self.attractor_sizes[0]) #initial winning region

    # get the shielded reachable set
    def shielded_reachable(self, state0=None):
        if state0 is None:
            state0 = self.mdp.state0
        self.automatonD.reset()
        self.automatonA.reset()
        s0 = tuple([*state0, self.automatonD.state, self.automatonA.state])

        if s0 not in self.safe_actions:
            raise KeyError(f"initial product state not in the game graph: {s0}")

        seen, stack = {s0}, [s0]
        while stack:
            v = stack.pop()
            for i in self.safe_actions[v]:
                nx = self.successors[v][i]
                if nx not in seen:
                    seen.add(nx)
                    stack.append(nx)
        self.reachable = seen
        return seen

    def winning_fraction(self):
        """WIN = |W| / (|S_x| - |U|): of the non-violating states, the fraction
        from which the defender can guarantee safety. Larger = more defensible."""
        total = len(self.mdp_product)
        U = self.attractor_sizes[0]
        return len(self.winning_region) / (total - U)

    def shell_steepness(self):
        """SEP = H / ln(k_max): normalized entropy of the shell (forcing-distance)
        distribution. Larger = graded boundary; near 0 = cliff. Larger is the
        more defensible reading, since concentration in a defensibility game
        occurs at the edge (shell 1).
        we report STP = 1 - SEP
        """
        import math
        total = self.attractor_sizes[-1] - self.attractor_sizes[0]
        num_shells = len(self.attractor_sizes) - 1
        if total == 0 or num_shells <= 1:
            return 1.0                      # single shell / no mass = cliff
        entropy = 0.0
        for i in range(1, len(self.attractor_sizes)):
            shell = self.attractor_sizes[i] - self.attractor_sizes[i - 1]
            if shell > 0:
                p_i = shell / total
                entropy -= p_i * math.log(p_i)
        return 1 - entropy / math.log(num_shells)

    def shield_latitude(self, domain):
        """SLT = sum|Safe| / sum|Act_D x V|: mean fraction of defender actions the
        shield PERMITS, averaged over `domain`. Larger = more room to act.
        `domain` must be given explicitly: the value depends strongly on it
        (full winning region 0.387, shielded-reachable 0.718 on the reference
        configuration).
        
        """
        n_actions = len(self.actionsD) * self.num_hosts
        permitted = total = 0
        for s in domain:
            if s[-3] != "D":
                continue
            total += n_actions
            permitted += len(self.safe_actions[s])
        return permitted / total if total else float("nan")

    def compute_metrics(self):
        shield_domain = self.shielded_reachable()
        return {"winning": self.winning_fraction(),
                "steepness":    self.shell_steepness(),
                "latitude":  self.shield_latitude(shield_domain)}





  


class ShieldedQlearning:
    def __init__(self, start_state, safe_actions, automatonD, automatonA, labeler, epsilon=0.1, alpha=0.9, discount=0.99, debug=debug) -> None:
        self.safe_actions = safe_actions
        self.labeler = labeler
        self.automatonD = automatonD
        self.automatonA = automatonA
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount = discount
        # 5 hosts,5 states and 5 actions each, Optimistic start with vals 5
        self.QvaluesD = np.full((5, 5, 5, 5, 5, 25), 5, dtype=float)
        self.QvaluesA = np.full((5, 5, 5, 5, 5, 15), 5, dtype=float)
        self.start_state = start_state
        self.host_states = ["C", "X", "D", "I", "Z"]
        self.host_states_idx = {h:i for i, h in enumerate(self.host_states)}
        self.debug = debug
        self.num_hosts = len(start_state)-1


    def states2Qidx(self, state):
        return [self.host_states_idx[h] for h in state[:-1]]
    
    def action2idx(self, action):
        # action = (type, host)
        idx = (self.num_hosts)*action[0] + action[1]
        return idx
    
    def idx2action(self, idx):
        action = (idx//(self.num_hosts), idx%(self.num_hosts))
        return action

    def shield(self, mdp_state):
        # self.automatonD.step(symbols)
        return self.safe_actions[tuple([*mdp_state + [self.automatonD.state] + [self.automatonA.state]])]

    def take_action(self, state):
        # state (h1, ..., hn, D)
        # argmax over safe actions
        safe_actions = self.shield(state)
        qstate = self.states2Qidx(state)
        if self.debug:
            print("Safe Actions:", safe_actions)
            print("Q states:", qstate)

        if np.random.rand() < self.epsilon:
            return np.random.choice(safe_actions)
        
        if state[-1] == "D":
            return  safe_actions[np.argmax([q for i,q in enumerate(self.QvaluesD[qstate[0], qstate[1], qstate[2], qstate[3], qstate[4]]) if i in safe_actions])]

        elif state[-1] == "A":
            # return np.random.choice(safe_actions)
            return  safe_actions[np.argmax([q for i,q in enumerate(self.QvaluesA[qstate[0], qstate[1], qstate[2], qstate[3], qstate[4]]) if i in safe_actions])]


    def update(self, state, action, reward, new_state):
        symbolsD = self.labeler.labeler_D(new_state, action, True)
        self.automatonD.step(symbolsD)
        if state[-1] == "A":
            symbolsA = self.labeler.labeler_A(new_state, action)
        else: 
            symbolsA = self.labeler.labeler_A(new_state, None)
        self.automatonA.step(symbolsA)
        
        
        safe_next_actions = self.shield(new_state)
        qstate = self.states2Qidx(state)
        next_qstate = self.states2Qidx(new_state)
        action = self.action2idx(action)
        if self.debug:
            print("Defense Symbols:", symbolsD)
            print("Automaton D:", self.automatonD.state)
            print("Automaton A:", self.automatonA.state) 
            print("Safe Next Actions", safe_next_actions)

        if state[-1] == "A": 
            qvals = self.QvaluesA[qstate[0], qstate[1], qstate[2], qstate[3], qstate[4], action]
            next_qvals = self.QvaluesD[next_qstate[0], next_qstate[1], next_qstate[2], next_qstate[3], next_qstate[4]]
            max_nextsafe_q = -1 * max([q for i,q in enumerate(next_qvals) if i in safe_next_actions ]) #Assuming defender will do the worst thing next
            self.QvaluesA[qstate[0], qstate[1], qstate[2], qstate[3], qstate[4], action] += self.alpha*(reward + self.discount*max_nextsafe_q - qvals)
            

        elif state[-1] == "D":
            # Shielded Update for defender
            qvals = self.QvaluesD[qstate[0], qstate[1], qstate[2], qstate[3], qstate[4], action]
            next_qvals = self.QvaluesA[next_qstate[0], next_qstate[1], next_qstate[2], next_qstate[3], next_qstate[4]]
            max_nextsafe_q = -1 * max([q for i,q in enumerate(next_qvals) if i in safe_next_actions ]) #Assuming attacker will do worst thing next
            self.QvaluesD[qstate[0], qstate[1], qstate[2], qstate[3], qstate[4], action] += self.alpha*(reward + self.discount*max_nextsafe_q - qvals)
            

    def reset(self):
        self.automatonD.reset()
        self.automatonA.reset()
           





   
            

def plot_mean_siqr_time(data):
    """
    data: list of lists, each inner list = samples at a time step
    """
    data = [np.array(d) for d in data]
    
    # Time axis
    t = np.arange(len(data))
    
    # Stats per time step
    mean = np.array([np.median(d) for d in data])
    q1 = np.array([np.percentile(d, 25) for d in data])
    q3 = np.array([np.percentile(d, 75) for d in data])
    
    # Plot mean line
    plt.plot(t, mean, label="Mean", linewidth=2)
    
    # Shaded IQR (Q1 to Q3)
    plt.fill_between(t, q1, q3, alpha=0.3, label="IQR (Q1–Q3)")


def plot_mean_std_time(data):
    """
    data: list of lists, each inner list = samples at a time step
    """
    data = [np.array(d) for d in data]
    
    # Time axis
    t = np.arange(len(data))
    
    # Stats per time step
    mean = np.array([np.mean(d) for d in data])
    std = np.array([np.std(d) for d in data])
    
    upper = mean + std
    lower = mean - std
    
    # Plot mean
    plt.plot(t, mean, linewidth=2, label="Mean")
    
    # Shaded std band
    plt.fill_between(t, lower, upper, alpha=0.3, label="±1 Std Dev")
    










    