import sys
import seaborn as sns
from scipy import stats
import os
import json
from datetime import datetime

from bastion5 import *
BASTION_IDX = 5
CASE_DESCRIPTION = "DB BYPASS ADDED + Double edges"

RESULTS_DIR = "../results"




def simulation(agent, max_iterations, adversary_respawn=0.1,  debug=False):
    cumulative_reward = [0]
    reward_time = []
    history_percent_clean = []
    average_reward_attacker = 0
    average_reward_defender= 0
    state = state0
    agent.reset()
    for i in range(max_iterations):
        if all(h == "C" for h in state[:-1]):
            if np.random.rand() < adversary_respawn:
                state[0] = "X"
                agent.reset() # automatons both reset
            
        
        a = agent.take_action(state)
        a_tuple = analyzer.idx2action(a)
        new_state, reward = analyzer.mdp.step(state, a_tuple)
        if debug:
            print("State", state)
            print("Action Idx", a)
            print("Action", a_tuple)
            print("Next State:", new_state)

        agent.update(state, a_tuple, reward, new_state)
        cumulative_reward.append(cumulative_reward[-1] + reward)
        reward_time.append(reward)

        

        percent_clean = sum(1 for h in state[:-1] if h=="C") / 5
        history_percent_clean.append(percent_clean)

        if state[-1] == "A":
            average_reward_attacker += reward/max_iterations
            # average_reward_defender -= reward/max_iterations
        elif state[-1] == "D":
            # average_reward_attacker -= reward/max_iterations
            average_reward_defender += reward/max_iterations
        

        if debug:
            print("Reward:", reward)
            print("Average Reward Attacker:", average_reward_attacker)
            print("Average Reward Defender:", average_reward_defender)
            print("="*20)
        
        if analyzer.in_attractor[analyzer.map_idx[(*new_state, agent.automatonD.state, agent.automatonA.state)]]==1:
            print("WHAAAAAAAAAAAAAAAT")
            sys.exit()

        state = new_state
    
    
    return reward_time, cumulative_reward, average_reward_attacker, average_reward_defender, np.mean(history_percent_clean)


def t_confidence_interval(data, confidence=0.95):
        """Return (mean, lower_bound, upper_bound) for a list of observations."""
        n = len(data)
        mean = np.mean(data)
        sem = np.std(data, ddof=1) / np.sqrt(n)   # standard error of the mean
        t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
        margin = t_crit * sem
        return mean, mean - margin, mean + margin



if __name__ == "__main__":

    DEBUG = False
    SIMULATION_EPISODES = 3000
    

    labeler = Labeler()
    analyzer = NetworkAnalyzer(topology, host_states, state0, mdp_transitions_D, mdp_transitions_A, actions_type_D, actions_type_A, automaton_D, automaton_A, qD_states, qA_states, labeler)
    _ , attractor_sizes = analyzer.compute_attractor()
    analyzer.compute_winning_region()

    print("=====Results=======")
    print(len(analyzer.attractor))
    print(len(analyzer.winning_region))

    analyzer.compute_safe_actions()

    metrics = analyzer.compute_metrics(0.95)

    shells = []
    for k in range(1, len(attractor_sizes)):
        shells.append(attractor_sizes[k] - attractor_sizes[k-1])
    total_states = len(analyzer.map_idx)  # |S×|
    initial_unsafe = attractor_sizes[0]   # |A0|
    attractor_final = attractor_sizes[-1] # |Attr*|
    winning_region = len(analyzer.winning_region)
    # attackability      = metrics["attackability"]
    # sinking_ratio      = metrics["sinking"]
    # shield_friction    = metrics["friction"]
    # attractor_steepness = metrics["steepness"]
    # mean_steps         = metrics["mean_steps"]

    safety_game_data = {
        "total_states":            total_states,
        "initial_unsafe_size":     initial_unsafe,
        "attractor_size":          attractor_final,
        "winning_region_size":     winning_region,
        "winning_region_pct":      round(winning_region / total_states * 100, 2),
        "attractor_shells":        shells,
        "attackability":           metrics["attackability"],
        "sinking_ratio":           metrics["sinking"],
        "shield_friction":         metrics["friction"],
        "attractor_steepness":     metrics["steepness"],
        "mean_steps_to_violation": metrics["mean_steps"],
    }


    # s = ('I', 'X', 'I', 'C', 'C', 'D', ('q0', 'safe'), 'q2')
    # # print("Successors:",analyzer.successors[s])
    # print("State:", s)
    # print("Safe Actions:", analyzer.safe_actions[s])
    # print("IN ATTRACTOR:", [analyzer.in_attractor[analyzer.map_idx[i]] for i in analyzer.successors[s]])
    # print("--------")
    
   
    # #9 (1, 4)
    # aidx = 5
    # a = (1,0)
    # print("Action Taken", a)
    # nstate = analyzer.successors[s][aidx]
    # print("State:", nstate)
    # print("Safe Actions:", analyzer.safe_actions[nstate])
    # print("IN ATTRACTOR OF NEXT STATE:", [analyzer.in_attractor[analyzer.map_idx[i]] for i in analyzer.successors[nstate]])
    # print("-----------")
    
    

    
    final_avRD_list = []   # Average Reward Defender (last 200)
    final_avRA_list = []   # Average Reward Attacker (last 200)
    final_clean_perc_list = []  # Defender Dominance Ratio % (last 200)

    robust_metrics = []
    all_reward_D = np.zeros((10, SIMULATION_EPISODES))
    all_reward_A = np.zeros((10, SIMULATION_EPISODES))
    all_clean_perc = np.zeros((10, SIMULATION_EPISODES))
    for i in range(10):
        np.random.seed(i+1)
        print("\n\n================= SIMULATION =====================")
        agent = ShieldedQlearning(state0, analyzer.safe_actions, automaton_D, automaton_A, labeler,
                                epsilon=0.5, alpha=0.05, discount=0.95, debug=DEBUG)
        average_reward_timeD = []
        average_reward_timeA = []
        clean_perc_time = []
        
        for episode in tqdm(range(SIMULATION_EPISODES)):
            reward_time, cumulative_reward, average_reward_A, average_reward_D, clean_perc = \
                simulation(agent, 1000, adversary_respawn=0.1, debug=DEBUG)
            
            if DEBUG:
                print("AVERAGE REWARD Attacker:", average_reward_A)
                print("AVERAGE REWARD Defender:", average_reward_D)
                print("AVERAGE CLEAN HOSTS %:", clean_perc)
                
            average_reward_timeD.append(average_reward_D)
            average_reward_timeA.append(average_reward_A)
            clean_perc_time.append(clean_perc)

            all_reward_D[i, episode] = average_reward_D
            all_reward_A[i, episode] = average_reward_A
            all_clean_perc[i, episode] = clean_perc

            agent.epsilon = max(agent.epsilon - 1/(2*SIMULATION_EPISODES), 0.1)
            agent.reset()
            # if episode == SIMULATION_EPISODES-2:
            #     agent.debug = True 
            #     DEBUG = True
        
        metrics = analyzer.compute_metrics(0.95)
        print(metrics)
        
        # Per-run summaries (using last 200 episodes)
        avRD = np.mean(average_reward_timeD[-200:])
        avRA = np.mean(average_reward_timeA[-200:])
        clean_perc_L200 = np.mean(clean_perc_time[-200:]) * 100  # in percent
        
        final_avRD_list.append(avRD)
        final_avRA_list.append(avRA)
        final_clean_perc_list.append(clean_perc_L200)
        
        print("Average Average Reward Defender:", sum(average_reward_timeD)/len(average_reward_timeD))
        print("Average Average Reward L200:", avRD)
        print("Average Average Reward Attacker:", sum(average_reward_timeA)/len(average_reward_timeA))
        print("Average Average Reward L200:", avRA)
        print("Defender Dominance Ratio %:", np.mean(clean_perc_time)*100, "%")
        print("Defender Dominance Ratio L200 %:", clean_perc_L200, "%")



    print("\n" + "="*50)
    print("T-TEST CONFIDENCE INTERVALS (95% CI, n=10)")
    print("="*50)

    mean_rd, ci_low_rd, ci_high_rd = t_confidence_interval(final_avRD_list)
    print(f"Avg Reward Defender (last 200): {mean_rd:.4f}  CI = [{ci_low_rd:.4f}, {ci_high_rd:.4f}]")

    mean_ra, ci_low_ra, ci_high_ra = t_confidence_interval(final_avRA_list)
    print(f"Avg Reward Attacker (last 200): {mean_ra:.4f}  CI = [{ci_low_ra:.4f}, {ci_high_ra:.4f}]")

    mean_clean, ci_low_clean, ci_high_clean = t_confidence_interval(final_clean_perc_list)
    print(f"Defender Dominance Ratio (last 200) %: {mean_clean:.2f}%  CI = [{ci_low_clean:.2f}%, {ci_high_clean:.2f}%]")

    # sys.exit()

    tag = f"bastion_{BASTION_IDX}"
    np.savez_compressed(
        os.path.join(RESULTS_DIR, f"{tag}_episodes.npz"),
        reward_D=all_reward_D,
        reward_A=all_reward_A,
        clean_perc=all_clean_perc,
    )

    summary = {
        "bastion_idx": BASTION_IDX,
        "case_description": CASE_DESCRIPTION,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_runs": 10,
            "simulation_episodes": SIMULATION_EPISODES,
            "episode_steps": 1000,
            "adversary_respawn": 0.1,
            "epsilon_init": 0.5, "epsilon_min": 0.1,
            "alpha": 0.05, "discount": 0.95,
        },
        "results": {
            "reward_defender": {"mean": mean_rd, "ci_low": ci_low_rd, "ci_high": ci_high_rd, "per_run": final_avRD_list},
            "reward_attacker": {"mean": mean_ra, "ci_low": ci_low_ra, "ci_high": ci_high_ra, "per_run": final_avRA_list},
            "defender_dominance_pct": {"mean": mean_clean, "ci_low": ci_low_clean, "ci_high": ci_high_clean, "per_run": final_clean_perc_list},
        },
    }
    summary["safety_game"] = safety_game_data

    with open(os.path.join(RESULTS_DIR, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {RESULTS_DIR}/{tag}_episodes.npz")
    print(f"Saved: {RESULTS_DIR}/{tag}_summary.json")
        


    print("\n\n")
    #Results 
    # ---- PLOTS (aggregate across all runs) ----
    tag = f"Bastion #{BASTION_IDX}"
    L200 = 200

    # Fig 1: Learning curves mean±std
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{tag} — Learning Curves (mean ± std, n=10)")
    for ax, data, label, color in [
        (axes[0], all_reward_D, "Defender Reward", "steelblue"),
        (axes[1], all_reward_A, "Attacker Reward", "orangered"),
        (axes[2], all_clean_perc * 100, "Clean Hosts %", "seagreen"),
    ]:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(mean, color=color, linewidth=1.2)
        ax.fill_between(range(SIMULATION_EPISODES), mean - std, mean + std, alpha=0.25, color=color)
        ax.axhline(0 if color!="seagreen" else 50, color="grey", ls="--", alpha=0.5)
        ax.set_xlabel("Episode"); ax.set_ylabel(label); ax.set_title(label)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"bastion_{BASTION_IDX}_learning_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # Fig 2: Convergence distributions (L200, all runs pooled)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{tag} — Convergence Distribution (L{L200}, pooled)")
    for ax, data, label, color in [
        (axes[0], all_reward_D[:, -L200:].flatten(), "Defender Reward", "steelblue"),
        (axes[1], all_reward_A[:, -L200:].flatten(), "Attacker Reward", "orangered"),
        (axes[2], all_clean_perc[:, -L200:].flatten() * 100, "Clean Hosts %", "seagreen"),
    ]:
        sns.histplot(data, bins=40, stat="density", kde=True, color=color, ax=ax, alpha=0.6)
        ax.axvline(np.mean(data), color="black", ls="--", linewidth=1.5, label=f"mean={np.mean(data):.3f}")
        ax.set_xlabel(label); ax.set_title(label); ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"bastion_{BASTION_IDX}_convergence_dist.png"), dpi=150, bbox_inches="tight")
    plt.show()

    # Fig 3: Per-run box plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{tag} — Per-Run Convergence (n=10)")
    for ax, data, label, color in [
        (axes[0], final_avRD_list, "Defender Reward (L200)", "steelblue"),
        (axes[1], final_avRA_list, "Attacker Reward (L200)", "orangered"),
        (axes[2], final_clean_perc_list, "DDR % (L200)", "seagreen"),
    ]:
        bp = ax.boxplot(data, patch_artist=True, widths=0.4)
        bp["boxes"][0].set_facecolor(color); bp["boxes"][0].set_alpha(0.5)
        ax.scatter([1]*len(data), data, color=color, zorder=3, s=40, alpha=0.8)
        ax.set_title(label); ax.set_xticks([])
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"bastion_{BASTION_IDX}_boxplots.png"), dpi=150, bbox_inches="tight")
    plt.show()

    
    
    
