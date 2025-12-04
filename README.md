# Phase 2: Term Project Mid-Status Report
--

## Group Members
- Archana Mannam
- Sai Supraja Ganta  
- Ajayi Oluwatomisin Raymond  

---

## Research Question
**How do Q-Learning and Value Iteration differ in convergence rate, cumulative reward, and policy optimality when applied to Gridworld and Pacman environments?**

---

## Introduction
Reinforcement Learning (RL) enables agents to learn optimal decision-making policies through interaction with an environment, receiving rewards or penalties for each action.  
This project compares two core RL algorithms—**Q-Learning** (model-free) and **Value Iteration** (model-based)—to evaluate their learning efficiency and convergence in two benchmark environments:

- **Gridworld:** Static, fully observable environment  
- **Pacman:** Dynamic, adversarial environment  

**Objective:** Identify which approach converges faster and produces more stable cumulative rewards, informing model-selection decisions in real-world autonomous systems.

---

## Related Literature
- **Sutton & Barto (2018)** – *Reinforcement Learning: An Introduction*: foundational treatment of Q-Learning and Value Iteration.  
- **UC Berkeley CS188 AI Projects** – standard Pacman/Gridworld framework for reinforcement-learning evaluation.  
- **Kaelbling et al. (1996)** – *Reinforcement Learning: A Survey*: comparative insights into model-based vs. model-free learning.  

Unlike prior analyses focusing on single algorithms, this study runs both under identical environmental conditions and metrics for direct quantitative comparison.

---

## Development Progress

### Environment Layer
- Gridworld and Pacman environments configured in Python using the Berkeley AI codebase.

### Agent Layer
- **Q-Learning Agent:** Implements Bellman update:  
# Reinforcement Learning Project: Gridworld & Pacman

## How to Run the Project

### Gridworld

**Value Iteration:**
python gridworld.py -a value -i 10

**Q-Learning:**
python gridworld.py -a q -k 50

### Pacman

**Value Iteration Agent:**
python pacman.py -p ValueIterationAgent -l smallClassic -a iterations=100

**Q-Learning Agent:**
python pacman.py -p PacmanQAgent -x 2000 -n 2100 -l smallClassic

---

## Example Outputs

### Gridworld (Value Iteration)
EPISODE 1 COMPLETE: RETURN WAS 0.59049
AVERAGE RETURNS FROM START STATE: 0.59049

Converges immediately due to full knowledge of the environment.

### Gridworld (Q-Learning)
EPISODE 1 COMPLETE: RETURN WAS 0.0133
...
EPISODE 50 COMPLETE

Gradual improvement; multiple episodes are required to approach the optimal policy.

### Pacman (Q-Learning)
Completed 2000 out of 2000 training episodes
Average Rewards last 100 episodes: -384.81

Reward improves progressively despite adversarial ghost behavior, showing learning under uncertainty.

---

## Observations
- Value Iteration converges almost instantly in environments with known transition probabilities (e.g., Gridworld).  
- Q-Learning requires many episodes but eventually approaches the optimal policy.  
- In dynamic environments like Pacman, Q-Learning steadily improves rewards, demonstrating effective learning under stochastic conditions.  
- Results align with theoretical expectations: model-based methods excel with complete models, while model-free methods adapt to complex, uncertain environments.

---

## Challenges and Adjustments
- Q-Learning reward oscillation: reduced α to 0.1 and increased ε to stabilize convergence.  
- Value Iteration overhead on large maps: introduced threshold-based convergence (Δ < 0.01).  
- Pacman adversarial variance: averaged multiple runs for smoother reward trends.  
- Environment compatibility: modularized shared state/action structures to run both agents seamlessly.

---

## Next Steps
- Test variable discount factors (γ = 0.8–0.99).  
- Visualize reward curves and policy convergence.  
- Integrate Deep-Q or neural network agents for comparison.  
- Compile the final report and presentation summarizing findings.

---

**Team Members:** Archana Mannam | Sai Supraja Ganta | Raymond Frimpong Amoateng
