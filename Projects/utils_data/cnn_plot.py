import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

    
def plot_parameters(W, number_rows=1, name="", i=0):
    W = W.data[:, i, :, :]
    n_filters = W.shape[0]
    w_min = W.min().item()
    w_max = W.max().item()

    colormap = 'viridis' 

    fig, axes = plt.subplots(number_rows, n_filters // number_rows, figsize=(12, 6))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    if n_filters // number_rows == 1:
        axes = [axes]
    axes_flat = axes.flat if isinstance(axes, (list, tuple, plt.Axes)) else [axes]
    axes_flat = axes.flat if hasattr(axes, 'flat') else axes


    for i, ax in enumerate(axes_flat):
        if i < n_filters:
            ax.set_xlabel("kernel:{0}".format(i + 1))
            
            im = ax.imshow(W[i, :], vmin=w_min, vmax=w_max, cmap=colormap) 
            
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            fig.delaxes(ax)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    plt.suptitle(name, fontsize=16)
    plt.show()
    
    
def plot_channels(W):
    n_out = W.shape[0]
    n_in = W.shape[1]

    n_images_to_plot = n_in 
    
    w_min = W.min().item()
    w_max = W.max().item()

    fig, axes = plt.subplots(1, n_images_to_plot, figsize=(n_images_to_plot * 2, 2))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    if n_images_to_plot == 1:
        axes = [axes]
        
    for i, ax in enumerate(axes):
        image_data = W[0, i, :, :] 
        ax.imshow(image_data, vmin=w_min, vmax=w_max, cmap='viridis')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Input {i+1}')

    plt.show()


def plot_activations(A, number_rows=1, name=""):
    A = A[0, :, :, :].detach().numpy()
    n_activations = A.shape[0]
    A_min = A.min().item()
    A_max = A.max().item()

    fig, axes = plt.subplots(number_rows, n_activations // number_rows, figsize=(15, 10))
    
    cmap = 'viridis' 
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    fig.suptitle(f"Activations for: {name}", fontsize=18, y=1.02)

    for i, ax in enumerate(axes.flat):
        if i < n_activations:
            im = ax.imshow(A[i, :], vmin=A_min, vmax=A_max, cmap=cmap)
            ax.set_title(f"Activation {i + 1}", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    
    plt.show()