import matplotlib.pyplot as plt
from IPython import display

# Enable interactive plotting in Jupyter notebooks
plt.ion()

# Function to plot scores and mean scores during training
def plot(scores, mean_scores):
    display.clear_output(wait=True)  # Clear the output before plotting
    display.display(plt.gcf())       # Display the current plot
    plt.clf()  # Clear the figure
    
    # Set plot title and axis labels
    plt.title('Training...')
    plt.xlabel('Number of Games')  # x-axis label
    plt.ylabel('Score')            # y-axis label
    
    # Plot the individual game scores
    plt.plot(scores)
    
    # Plot the mean scores over time
    plt.plot(mean_scores)
    
    # Set y-axis limits
    plt.ylim(ymin=0)  # Ensure y-axis starts at zero
    
    # Add text to display the latest score
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    
    # Add text to display the latest mean score
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    
    plt.show(block=False)  # Show the plot without blocking code execution
    
    plt.pause(0.1)  # Pause briefly to allow the plot to update
