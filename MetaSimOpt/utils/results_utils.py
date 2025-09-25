from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis


def print_training(losses, val_losses, scores = None, val_scores = None):
    
    print("\nCOMPLETED TRAINING!!!")

    if val_losses:
        best_epoch = val_losses.index(min(val_losses))
    else:
        best_epoch = losses.index(min(losses))

    print(f'Best epoch --> {best_epoch+1}')
    
    loss = losses[best_epoch]
    print(f'Training loss --> {loss}')
    
    if val_losses:
        val_loss = val_losses[best_epoch]
        print(f'Validation loss --> {val_loss}')

    if scores:
        score = scores[best_epoch]
        print(f'Training score --> {score}')
    
    if val_scores:
        val_score = val_scores[best_epoch]
        print(f'Validation score --> {val_score}')


def plot_training(losses, val_losses, scores = None, val_scores = None):

    n_epochs = len(losses)
    
    validation = True if val_losses else False

    if scores:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    else:
        fig, (ax1) = plt.subplots(1, 1, figsize=(10, 4))

    # Plot losses
    ax1.plot(range(1, n_epochs + 1), losses, label='Training Loss')
    if validation:
        ax1.plot(range(1, n_epochs + 1), val_losses, label='Validation Loss')
    ax1.legend()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs. Epoch')

    if scores:
        # Plot scores
        ax2.plot(range(1, n_epochs + 1), scores, label='Training Score')
        if validation:
            ax2.plot(range(1, n_epochs + 1), val_scores, label='Validation Score')
        ax2.legend()
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('Score vs. Epoch')

    plt.tight_layout()
    plt.show()


def compute_residual_stats(predictions, targets):
        
    if len(targets.shape) != len(predictions.shape):
        n = predictions.shape[-1]
        targets = np.repeat(targets[:, :, np.newaxis], n, axis=2)
        
    residuals = (predictions - targets).flatten()
    residuals_stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'skew': skew(residuals),
        'kurtosis': kurtosis(residuals)
    }

    return residuals, residuals_stats


def print_residuals(data):
    
    print("\nResidual statistics:")
    for k, v in data.items():
        print(f"{k.capitalize()}: {v:.4f}")


def plot_residuals(residuals, predictions):

    residuals = residuals.flatten().squeeze()
    predictions = predictions.flatten().squeeze()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Predictions vs residuals plot
    ax1.scatter(predictions, residuals, color = "steelblue", s = 20)
    ax1.axhline(y=0, color="orange", linestyle='--', linewidth=2.5)
    ax1.axhline(y=np.mean(residuals), color="cyan", linestyle='--', linewidth=2.5)
    ax1.set_title(f'Predictions vs Residuals')
    ax1.set_xlabel(f'Predictions [s]')
    ax1.set_ylabel('Residuals [s]')
    ax1.spines['top'].set_color('none')
    ax1.spines['right'].set_color('none')
    ax1.grid(True, linestyle='--', alpha=0.5)  # Add grid
    ax1.tick_params(axis='both', which='major')  # Set tick font size

    # Residual distribution plot
    ax2.hist(residuals, bins=50, color = "steelblue")
    ax2.set_title(f'Residual Distribution - Output')
    ax2.set_xlabel('Residuals [s]')
    ax2.set_ylabel('Frequency [#]')
    ax2.spines['top'].set_color('none')
    ax2.spines['right'].set_color('none')
    ax2.grid(True, linestyle='--', alpha=0.5)  # Add grid
    ax2.tick_params(axis='both', which='major')  # Set tick font size

    # Number of predictions vs. residuals plot
    ax3.scatter(np.arange(len(residuals)), residuals, color = "steelblue", s = 20)
    ax3.axhline(y=0, color="orange", linestyle='--', linewidth=2.5)
    ax3.axhline(y=np.mean(residuals), color="cyan", linestyle='--', linewidth=2.5)
    ax3.set_title(f'Number of Predictions vs. Residuals')
    ax3.set_xlabel('Order [#]')
    ax3.set_ylabel('Residuals [s]')
    ax3.spines['top'].set_color('none')
    ax3.spines['right'].set_color('none')
    ax3.grid(True, linestyle='--', alpha=0.5)  # Add grid
    ax3.tick_params(axis='both', which='major')  # Set tick font size

    plt.tight_layout()
    plt.show()


def compute_metrics(targets, predictions, metrics = ["mae"]):

    if len(targets.shape) != len(predictions.shape):
        n = predictions.shape[-1]
        targets = np.repeat(targets[:, :, np.newaxis], n, axis=2)

    supported_metrics = ["mse", "mae", "mape"]
    for m in metrics:
        if m not in supported_metrics:
            raise ValueError(f"Metric {m} not supported. Choose among {supported_metrics}")
        
    def mse(targets,predictions):
        return np.mean((targets - predictions)**2)

    def mae(targets,predictions):
        return np.mean(np.abs(targets - predictions))

    def mape(targets,predictions):
        return np.mean(np.abs((targets - predictions) / targets) * 100)  # in percentage
    
    map_metrics = {
        "mae" : mae,
        "mape" : mape,
        "mse" : mse
    }

    results = []
    for m in metrics:
        results.append(map_metrics[m](targets=targets, predictions=predictions))
    
    return results


def plot_opt(best_fitness_per_iteration, all_fitness_per_iteration, algorithm = "Algorithm", plot_prog = True, plot_var = True):
    # Calcolare la media del fitness per generazione
    mean_fitness_per_generation = [np.mean(fitness) for fitness in all_fitness_per_iteration]
    var_fitness_per_generation = [np.var(fitness) for fitness in all_fitness_per_iteration]

    if plot_prog:
        # Plot fitness progression
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(best_fitness_per_iteration)), best_fitness_per_iteration, label='Best Fitness', color='blue')
        plt.plot(range(len(mean_fitness_per_generation)), mean_fitness_per_generation, label='Mean Fitness', color='orange')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(f'{algorithm} - Fitness Progression')
        plt.legend()
        plt.grid(True)
        plt.show()

    if plot_var:
        # Plot population variability
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(best_fitness_per_iteration)), var_fitness_per_generation, label='Var Fitness', color='green')
        plt.xlabel('Generation')
        plt.ylabel('Var fitness')
        plt.title(f'{algorithm} - Var Fitness Progression')
        plt.legend()
        plt.grid(True)
        plt.show()