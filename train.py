import gymnasium as gym
import numpy as np
import imageio
import random
import json
import datetime
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save
from pathlib import Path
import pickle
from tqdm import tqdm

# Initialize environment
env_id = "Taxi-v3"
env = gym.make(env_id)

state_size = env.observation_space.n
action_size = env.action_space.n

# Hyperparameters
number_of_episodes = 300000
learning_rate = 0.1  # Initial learning rate
learning_rate_min = 0.01  # Minimum learning rate
learning_rate_decay = 0.999999  # Decay over time
epsilon_decay = 0.999995
discount_factor = 0.995
epsilon = 1.0
epsilon_min = 0.1

eval_seed = [
    16,
    54,
    165,
    177,
    191,
    191,
    120,
    80,
    149,
    178,
    48,
    38,
    6,
    125,
    174,
    73,
    50,
    172,
    100,
    148,
    146,
    6,
    25,
    40,
    68,
    148,
    49,
    167,
    9,
    97,
    164,
    176,
    61,
    7,
    54,
    55,
    161,
    131,
    184,
    51,
    170,
    12,
    120,
    113,
    95,
    126,
    51,
    98,
    36,
    135,
    54,
    82,
    45,
    95,
    89,
    59,
    95,
    124,
    9,
    113,
    58,
    85,
    51,
    134,
    121,
    169,
    105,
    21,
    30,
    11,
    50,
    65,
    12,
    43,
    82,
    145,
    152,
    97,
    106,
    55,
    31,
    85,
    38,
    112,
    102,
    168,
    123,
    97,
    21,
    83,
    158,
    26,
    80,
    63,
    5,
    81,
    32,
    11,
    28,
    148,
]

# Initialize Q-table
qtable = np.zeros((state_size, action_size))

# Training the agent with a progress bar
for episode_nr in tqdm(
    range(number_of_episodes), desc="Training Episodes", unit="episode"
):
    learning_rate = max(learning_rate_min, learning_rate * learning_rate_decay)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    current_state, _ = env.reset()
    done = False
    total_reward = 0  # Initialize total_reward for monitoring

    while not done:
        # Epsilon-greedy strategy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qtable[current_state, :])

        next_state, reward, terminated, truncated, info = env.step(action)

        # Update Q-value
        future_q_value = np.max(qtable[next_state, :])
        current_q = qtable[current_state, action]
        new_q = current_q + learning_rate * (
            reward + discount_factor * future_q_value - current_q
        )
        qtable[current_state, action] = new_q
        current_state = next_state

        total_reward += reward  # Accumulate reward
        done = terminated or truncated  # Update 'done' variable

    if episode_nr % 10000 == 0:
        print(f"Episode {episode_nr}: Total Reward = {total_reward}")

# Save Q-table
np.save("TaxiV3_qtable.npy", qtable)

# Record the video
frames = []
env = gym.make(
    env_id,
    render_mode="rgb_array",
)
current_state, _ = env.reset()
done = False

while not done:
    frames.append(env.render())
    action = np.argmax(qtable[current_state, :])
    next_state, reward, done, truncated, _ = env.step(action)
    current_state = next_state

env.close()

# Save frames as a video
video_filename = "replay.mp4"
with imageio.get_writer(video_filename, fps=1) as video:
    for frame in frames:
        video.append_data(frame)

print(f"Video saved as {video_filename}")

# Prepare model data for Hugging Face Hub
model = {
    "env_id": env_id,
    "max_steps": 200,
    "n_training_episodes": number_of_episodes,
    "n_eval_episodes": 100,
    "eval_seed": eval_seed,
    "learning_rate": learning_rate,
    "gamma": discount_factor,
    "qtable": qtable,
}


def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(Qtable[state][:])

    return action


# Define helper function to evaluate agent's performance
def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param max_steps: Maximum number of steps per episode
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param Q: The Q-table
    :param seed: The evaluation seed array (for taxi-v3)
    """
    episode_rewards = []
    for episode in tqdm(range(n_eval_episodes)):
        state, info = env.reset(seed=seed[episode])
        truncated = False
        terminated = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = greedy_policy(Q, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return mean_reward, std_reward


# Define helper function to record a video of the agent's performance
def record_video(env, qtable, out_directory, fps=1):
    images = []
    done = False
    state, _ = env.reset(seed=random.randint(0, 500))
    img = env.render()
    images.append(img)
    while not done:
        action = np.argmax(qtable[state][:])
        state, _, done, _, _ = env.step(action)
        img = env.render()
        images.append(img)
    imageio.mimsave(out_directory, [np.array(img) for img in images], fps=fps)


def push_to_hub(repo_id, model, env, video_fps=1, local_repo_path="hub"):
    """
    Evaluate, Generate a video and Upload a model to Hugging Face Hub.
    This method does the complete pipeline:
    - It evaluates the model
    - It generates the model card
    - It generates a replay video of the agent
    - It pushes everything to the Hub

    :param repo_id: repo_id: id of the model repository from the Hugging Face Hub
    :param env
    :param video_fps: how many frame per seconds to record our video replay
    (with taxi-v3 and frozenlake-v1 we use 1)
    :param local_repo_path: where the local repository is
    """
    _, repo_name = repo_id.split("/")

    eval_env = env
    api = HfApi()

    # Step 1: Create the repo
    repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
    )

    # Step 2: Download files
    repo_local_path = Path(snapshot_download(repo_id=repo_id))

    # Step 3: Save the model
    if env.spec.kwargs.get("map_name"):
        model["map_name"] = env.spec.kwargs.get("map_name")
        if env.spec.kwargs.get("is_slippery", "") is False:
            model["slippery"] = False

    # Pickle the model
    with open((repo_local_path) / "q-learning.pkl", "wb") as f:
        pickle.dump(model, f)

    # Step 4: Evaluate the model and build JSON with evaluation metrics
    mean_reward, std_reward = evaluate_agent(
        eval_env,
        model["max_steps"],
        model["n_eval_episodes"],
        model["qtable"],
        model["eval_seed"],
    )

    evaluate_data = {
        "env_id": model["env_id"],
        "mean_reward": mean_reward,
        "n_eval_episodes": model["n_eval_episodes"],
        "eval_datetime": datetime.datetime.now().isoformat(),
    }

    # Write a JSON file called "results.json" that will contain the
    # evaluation results
    with open(repo_local_path / "results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    # Step 5: Create the model card
    env_name = model["env_id"]
    if env.spec.kwargs.get("map_name"):
        env_name += "-" + env.spec.kwargs.get("map_name")

    if env.spec.kwargs.get("is_slippery", "") is False:
        env_name += "-" + "no_slippery"

    metadata = {}
    metadata["tags"] = [
        env_name,
        "q-learning",
        "reinforcement-learning",
        "custom-implementation",
    ]

    # Add metrics
    eval = metadata_eval_result(
        model_pretty_name=repo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_name,
        dataset_id=env_name,
    )

    # Merges both dictionaries
    metadata = {**metadata, **eval}

    model_card = f"""
  # **Q-Learning** Agent playing1 **{env_id}**
  This is a trained model of a **Q-Learning** agent playing **{env_id}** .

  ## Usage

  ```python

  model = load_from_hub(repo_id="{repo_id}", filename="q-learning.pkl")

  # Don't forget to check if you need to add additional attributes (is_slippery=False etc)
  env = gym.make(model["env_id"])
  ```
  """

    mean_rewards = []
    for _ in range(5):  # Run 5 evaluations
        mean_reward, std_reward = evaluate_agent(
            env,
            model["max_steps"],
            model["n_eval_episodes"],
            model["qtable"],
            model["eval_seed"],
        )
        mean_rewards.append(mean_reward)
    print(f"Average Mean Reward over 5 runs: {np.mean(mean_rewards)}")

    readme_path = repo_local_path / "README.md"
    readme = ""
    print(readme_path.exists())
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    else:
        readme = model_card

    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)

    # Step 6: Record a video
    video_path = repo_local_path / "replay.mp4"
    record_video(env, model["qtable"], video_path, video_fps)

    # Step 7. Push everything to the Hub
    api.upload_folder(
        repo_id=repo_id,
        folder_path=repo_local_path,
        path_in_repo=".",
    )

    print("Your model is pushed to the Hub. You can view your model here: ", repo_url)


# Set Hugging Face repo details and push the model
username = "pkalkman"  # Replace with your Hugging Face username
repo_name = "Taxi-v3"
repo_id = f"{username}/{repo_name}"

push_to_hub(repo_id=repo_id, model=model, env=env)
