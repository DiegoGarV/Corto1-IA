import gymnasium as gym
import numpy as np
import random

def train_q_learning(env, episodes=10000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.001):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    Q = np.zeros((state_size, action_size))
    
    print("Agente entrenando...")
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explorar
            else:
                action = np.argmax(Q[state, :])  # Explotar mejor acción
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]))
            state = next_state
        
        # Reducir epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    print("Agente listo")
    return Q

def evaluate_agent(Q, env_name="FrozenLake-v1", episodes=10):
    env = gym.make(env_name, is_slippery=True, render_mode="human")
    wins = 0
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        print(f"Episodio {episode + 1}:")
        
        while not done:
            env.render()
            action = np.argmax(Q[state, :])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if reward == 1:
                wins += 1
        
        print("Fin del episodio\n")
    
    print(f"El agente ganó {wins}/{episodes} episodios")
    env.close()

# Entrenar el agente en un entorno por defecto
env_train = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)
Q = train_q_learning(env_train)
env_train.close()

# Evaluar el agente en diferentes mapas
evaluate_agent(Q, "FrozenLake-v1")