import torch
from torchviz import make_dot
from model import Linear_QNet

input_size = 11
hidden_size = 256
output_size = 3

# Create a dummy input with the correct size
x = torch.randn(1, input_size)

# Create the model instance and pass the dummy input through it
model = Linear_QNet(input_size, hidden_size, output_size)
y = model(x)

# Generate the visualization
dot = make_dot(y, params=dict(model.named_parameters()))

# Save the visualization to a PNG file
dot.render("model_architecture", format="png", cleanup=True)
