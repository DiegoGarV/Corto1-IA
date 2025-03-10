import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt


def train_q_learning(
    env,
    episodes=15000,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=1.0,
    epsilon_decay=0.0001,
):
    state_size = env.observation_space.n
    action_size = env.action_space.n
    Q = np.zeros((state_size, action_size))

    print("Agente entrenando...")
    rewards_per_episode = np.zeros(episodes)

    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            Q[state, action] = (1 - learning_rate) * Q[
                state, action
            ] + learning_rate * (reward + discount_factor * np.max(Q[next_state, :]))
            state = next_state

        epsilon = max(epsilon - epsilon_decay, 0)

        if reward == 1:
            rewards_per_episode[episode] = 1

    print("Agente listo")

    # Gráfica de desempeño
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100) : (t + 1)])
    plt.plot(sum_rewards)
    plt.xlabel("Episodios")
    plt.ylabel("Victorias en ventana de 100 episodios")
    plt.title("Desempeño del Agente en FrozenLake")
    plt.savefig("frozen_lake_training.png")
    plt.show()

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


# Entrenar el agente
env_train = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)
Q = train_q_learning(env_train)
env_train.close()

# Evaluar el agente en diferentes mapas
evaluate_agent(Q, "FrozenLake-v1")
