import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union

def plot_metrics(loss_history: Union[List[float], np.ndarray], accuracy_history: Union[List[float], np.ndarray]):
    """
    Generates a combined plot of training loss and validation accuracy.

    Parameters:
        loss_history: A list or array of total training loss values.
        accuracy_history: A list or array of validation accuracy values.
        
    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    
    # Create a figure and primary axis (ax1) for the loss plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Define colors for clarity
    loss_color = 'tab:red'
    accuracy_color = 'tab:blue'

    # Plot loss on the primary y-axis
    ax1.set_xlabel('Iteration/Epoch')
    ax1.set_ylabel('Total Loss', color=loss_color)
    ax1.plot(loss_history, color=loss_color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=loss_color)
    ax1.grid(True, axis='x', linestyle='--', alpha=0.6)

    # Create a secondary y-axis (ax2) sharing the same x-axis
    ax2 = ax1.twinx()
    
    # Plot accuracy on the secondary y-axis
    ax2.set_ylabel('Accuracy', color=accuracy_color)
    ax2.plot(accuracy_history, color=accuracy_color, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=accuracy_color)
    ax2.set_ylim(0, 1.1)

    # Add a title and ensure the layout is tight
    plt.title('Training Loss and Validation Accuracy Over Iterations')
    fig.tight_layout() 
    
    # Display the plot
    plt.show()
