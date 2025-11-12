import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Any, Dict, Optional, Iterable 
from sklearn.metrics import classification_report

# Function to plot confusion matrix
def plot_confusion_matrix(
    y_true: Iterable[Any], 
    y_pred: Iterable[Any], 
    class_names: Iterable[Any],
    title: str = 'Confusion Matrix', 
    normalize: bool = False, 
    figsize: Tuple[int, int] = (10, 8), 
    cmap: str = 'viridis' 
):
    # Input validation:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not class_names:
        raise ValueError("class_names cannot be an empty list.")


    bins = len(class_names)
    # Make a 2D histogram from the test and result arrays
    cm, xe, ye = np.histogram2d(y_true, y_pred, bins)

    cm = pd.DataFrame(cm.astype(int), index=class_names, columns=class_names)
    
    if normalize:
        # Normalization logic remains the same
        cm_normalized: np.ndarray = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm_normalized, nan=0.0) 
        fmt: str = '.2%' 
        title = title + ' (Normalized by True Label)' if title else 'Normalized Confusion Matrix'
    else:
        fmt = 'd' 

    # Use the filtered class names for the DataFrame index/columns
    cm_df: pd.DataFrame = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    #  Plotting
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_df, 
        annot=True,     
        fmt=fmt,        
        cmap=cmap,      
        cbar=True,      
        linecolor='white' 
    )
    
    plt.title(title, fontsize=16, pad=20) 
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    
    plt.tight_layout()
    plt.show();
    plt.close()
    

# Function to display classification report
def display_classification_report(y_true, y_pred, target_names):
    """Prints a neatly formatted classification report."""
    
    # Generate the report string
    clr = classification_report(y_true, y_pred, target_names=target_names)
    
    # Print with improved title formatting
    print(f"{'Classification Report':^65}") # Center the title within a 65-char width
    print(65 * '-') # Manual line
    print(clr)