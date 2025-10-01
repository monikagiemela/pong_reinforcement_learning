import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import RecordVideo
import time

# --- 1. Model Definition (Must be identical to the one used for evaluation) ---
class DuelingDQN(nn.Module):
    """Convolutional neural network for the Atari games."""
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        std = math.sqrt(2.0 / (4 * 84 * 84))
        nn.init.normal_(self.conv1.weight, 0.0, std)
        self.conv1.bias.data.fill_(0.0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        std = math.sqrt(2.0 / (32 * 4 * 8 * 8))
        nn.init.normal_(self.conv2.weight, 0.0, std)
        self.conv2.bias.data.fill_(0.0)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        std = math.sqrt(2.0 / (64 * 32 * 4 * 4))
        nn.init.normal_(self.conv3.weight, 0.0, std)
        self.conv3.bias.data.fill_(0.0)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        std = math.sqrt(2.0 / (64 * 64 * 3 * 3))
        nn.init.normal_(self.fc1.weight, 0.0, std)
        self.fc1.bias.data.fill_(0.0)

        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, num_actions)

    def forward(self, x):
        """Forward pass of the neural network with some inputs."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1))) # Flatten input.
        V = self.V(x)
        A = self.A(x)
        return V + (A - A.mean(dim=1, keepdim=True))

# --- 2. Video Recording Function ---
def record_agent_video(model_path, video_folder='videos'):
    """Loads a trained agent and records a video of it playing one episode."""
    
    # --- Environment Setup ---
    # Step 1: Create the base environment
    env = gym.make('ALE/Pong-v5', render_mode='rgb_array', frameskip=1)
    
    # Step 2: Wrap the environment to record a video
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: e == 0, name_prefix="pong-agent")

    # Step 3: Apply the same preprocessing wrappers as in training
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=False, terminal_on_life_loss=False)
    env = FrameStackObservation(env, stack_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_actions = env.action_space.n
    print(f"Recording video using device: {device}")

    # Instantiate the model
    model = DuelingDQN(num_actions).to(device)

    # Load the trained weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['main_nn_state_dict'])
        print(f"Successfully loaded model weights from '{model_path}'.")
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        env.close()
        return

    model.eval()

    # Run one episode to record it
    state, info = env.reset()
    done = False
    print("Starting video recording for one episode...")
    while not done:
        # Prepare state tensor
        state_tensor = torch.from_numpy(np.array(state)).unsqueeze(0).to(device, dtype=torch.float32)
        state_tensor /= 255.0

        # Get action from the model
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        # Perform action
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()
    print(f"Video saved in the '{video_folder}' directory.")

# --- 3. Main Execution Block ---
if __name__ == "__main__":
    MODEL_CHECKPOINT_PATH = "checkpoint_ep5000_20251001_121214.pth"
    record_agent_video(model_path=MODEL_CHECKPOINT_PATH)