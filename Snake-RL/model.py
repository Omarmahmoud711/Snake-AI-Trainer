import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Define a simple neural network for Q-learning with two linear layers
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)  # First linear layer
        self.linear2 = nn.Linear(hidden_size, output_size) # Second linear layer

    # Define the forward pass for the neural network
    def forward(self, x):
        x = F.relu(self.linear1(x))  # Apply ReLU activation function
        x = self.linear2(x)          # Output from the second linear layer
        return x

    # Save the model to a file
    def save(self, file_name='model.pth'):
        model_folder_path = './model'  # Directory for saving the model
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)  # Create directory if it doesn't exist
        file_name = os.path.join(model_folder_path, file_name)  # Full file path
        torch.save(self.state_dict(), file_name)  # Save model parameters


# Trainer for Q-learning, manages training steps and optimization
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr               # Learning rate for optimizer
        self.gamma = gamma         # Discount factor for Q-learning
        self.model = model         # The model being trained
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)  # Adam optimizer
        self.criterion = nn.MSELoss()  # Mean squared error loss function

    # Single training step for Q-learning
    def train_step(self, state, action, reward, next_state, done):
        # Convert inputs to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Ensure inputs have correct shape, adding batch dimension if needed
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)  # (1, x)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Get predicted Q values for current state
        pred = self.model(state)

        # Clone predictions to create targets for training
        target = pred.clone()
        # Update target values based on Q-learning rules
        for idx in range(len(done)):
            Q_new = reward[idx]  # Default to reward for terminal states
            if not done[idx]:
                # If not terminal, calculate new Q-value with discounting
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            # Update the predicted action with the new Q-value
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Compute loss between predicted and target Q-values
        self.optimizer.zero_grad()  # Zero out gradients
        loss = self.criterion(target, pred)  # Calculate loss
        loss.backward()  # Backpropagation to compute gradients
        self.optimizer.step()  # Update model parameters with optimizer
