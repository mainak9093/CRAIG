"""
CRAIG Training Results Visualization Script
Prints and plots the training results from CRAIG experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def load_latest_results(folder='d:/tmp'):
    """Load the most recent results file from the specified folder."""
    pattern = os.path.join(folder, 'cifar10_*.npz')
    files = glob.glob(pattern)
    
    if not files:
        print(f"No results files found in {folder}")
        print("Make sure you have run the training script first!")
        return None
    
    # Get the most recent file
    latest_file = max(files, key=os.path.getctime)
    print(f"Loading results from: {latest_file}")
    
    return np.load(latest_file)

def print_results(data):
    """Print all available metrics from the results."""
    print("\n" + "="*60)
    print("CRAIG TRAINING RESULTS")
    print("="*60)
    
    print(f"\nAvailable metrics: {data.files}")
    
    # Print key metrics
    if 'test_acc' in data:
        test_acc = data['test_acc']
        print(f"\n�� TEST ACCURACY:")
        print(f"   Shape: {test_acc.shape}")
        print(f"   Min: {np.min(test_acc):.2f}%")
        print(f"   Max: {np.max(test_acc):.2f}%")
        print(f"   Mean (final): {np.mean(test_acc[:, -1]):.2f}%")
        if test_acc.shape[0] > 1:
            print(f"   Std (final): {np.std(test_acc[:, -1]):.2f}%")
    
    if 'train_acc' in data:
        train_acc = data['train_acc']
        print(f"\n�� TRAIN ACCURACY:")
        print(f"   Shape: {train_acc.shape}")
        print(f"   Min: {np.min(train_acc):.2f}%")
        print(f"   Max: {np.max(train_acc):.2f}%")
        print(f"   Mean (final): {np.mean(train_acc[:, -1]):.2f}%")
    
    if 'test_loss' in data:
        test_loss = data['test_loss']
        print(f"\n�� TEST LOSS:")
        print(f"   Shape: {test_loss.shape}")
        print(f"   Min: {np.min(test_loss):.4f}")
        print(f"   Max: {np.max(test_loss):.4f}")
        print(f"   Mean (final): {np.mean(test_loss[:, -1]):.4f}")
    
    if 'train_loss' in data:
        train_loss = data['train_loss']
        print(f"\n�� TRAIN LOSS:")
        print(f"   Shape: {train_loss.shape}")
        print(f"   Min: {np.min(train_loss):.4f}")
        print(f"   Max: {np.max(train_loss):.4f}")
        print(f"   Mean (final): {np.mean(train_loss[:, -1]):.4f}")
    
    if 'not_selected' in data:
        not_selected = data['not_selected']
        print(f"\n�� DATA COVERAGE (not selected %):")
        print(f"   Shape: {not_selected.shape}")
        print(f"   Min: {np.min(not_selected):.2f}%")
        print(f"   Max: {np.max(not_selected):.2f}%")
        print(f"   Final coverage: {100 - np.mean(not_selected[:, -1]):.2f}%")
    
    if 'best_g' in data:
        print(f"\n⚙️ BEST HYPERPARAMETERS:")
        print(f"   Learning rate (best_g): {data['best_g']}")
        print(f"   Decay (best_b): {data['best_b']}")
    
    if 'train_time' in data:
        train_time = data['train_time']
        print(f"\n⏱️ TRAINING TIME:")
        print(f"   Total time: {np.sum(train_time):.2f} seconds")
        print(f"   Avg per epoch: {np.mean(train_time):.2f} seconds")
    
    if 'grd_time' in data:
        grd_time = data['grd_time']
        print(f"\n⏱️ GREEDY SELECTION TIME:")
        print(f"   Total time: {np.sum(grd_time):.2f} seconds")
    
    print("\n" + "="*60)

def plot_results(data, save_path=None):
    """Create comprehensive plots of the training results."""
    
    # Set up the figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('CRAIG Training Results', fontsize=16, fontweight='bold')
    
    # Determine number of runs
    n_runs = data['test_acc'].shape[0] if 'test_acc' in data else 1
    epochs = data['test_acc'].shape[1] if 'test_acc' in data else 1
    
    # Create subplot grid
    if 'train_loss' in data:
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    else:
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Accuracy over epochs
    ax1 = fig.add_subplot(gs[0, 0])
    if 'test_acc' in data:
        for i in range(n_runs):
            ax1.plot(range(epochs), data['test_acc'][i], alpha=0.5, label=f'Run {i+1}' if n_runs <= 5 else '')
        mean_acc = np.mean(data['test_acc'], axis=0)
        ax1.plot(range(epochs), mean_acc, 'b-', linewidth=2, label=f'Mean (n={n_runs})')
        ax1.fill_between(range(epochs), 
                         mean_acc - np.std(data['test_acc'], axis=0),
                         mean_acc + np.std(data['test_acc'], axis=0),
                         alpha=0.2, label='±1 Std')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Test Accuracy Over Epochs')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss over epochs
    ax2 = fig.add_subplot(gs[0, 1])
    if 'test_loss' in data:
        for i in range(n_runs):
            ax2.plot(range(epochs), data['test_loss'][i], alpha=0.5, label=f'Run {i+1}' if n_runs <= 5 else '')
        mean_loss = np.mean(data['test_loss'], axis=0)
        ax2.plot(range(epochs), mean_loss, 'r-', linewidth=2, label=f'Mean (n={n_runs})')
        ax2.fill_between(range(epochs),
                        mean_loss - np.std(data['test_loss'], axis=0),
                        mean_loss + np.std(data['test_loss'], axis=0),
                        alpha=0.2, label='±1 Std')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Test Loss Over Epochs')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training vs Test accuracy (if available)
    ax3 = fig.add_subplot(gs[0, 2])
    if 'train_acc' in data and 'test_acc' in data:
        ax3.plot(range(epochs), np.mean(data['train_acc'], axis=0), 'g-', 
                 linewidth=2, label='Train Accuracy')
        ax3.plot(range(epochs), np.mean(data['test_acc'], axis=0), 'b-', 
                 linewidth=2, label='Test Accuracy')
        ax3.fill_between(range(epochs),
                        np.mean(data['train_acc'], axis=0) - np.std(data['train_acc'], axis=0),
                        np.mean(data['train_acc'], axis=0) + np.std(data['train_acc'], axis=0),
                        alpha=0.2, color='green')
        ax3.fill_between(range(epochs),
                        np.mean(data['test_acc'], axis=0) - np.std(data['test_acc'], axis=0),
                        np.mean(data['test_acc'], axis=0) + np.std(data['test_acc'], axis=0),
                        alpha=0.2, color='blue')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Train vs Test Accuracy')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Data coverage (not selected %)
    ax4 = fig.add_subplot(gs[1, 0])
    if 'not_selected' in data:
        for i in range(n_runs):
            ax4.plot(range(epochs), data['not_selected'][i], alpha=0.5)
        ax4.plot(range(epochs), np.mean(data['not_selected'], axis=0), 'purple', 
                 linewidth=2, label='Mean')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Not Selected (%)')
        ax4.set_title('Data Coverage Over Epochs\n(Lower = More Data Selected)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Train vs Test Loss (if available)
    ax5 = fig.add_subplot(gs[1, 1])
    if 'train_loss' in data and 'test_loss' in data:
        ax5.plot(range(epochs), np.mean(data['train_loss'], axis=0), 'g-', 
                 linewidth=2, label='Train Loss')
        ax5.plot(range(epochs), np.mean(data['test_loss'], axis=0), 'r-', 
                 linewidth=2, label='Test Loss')
        ax5.fill_between(range(epochs),
                        np.mean(data['train_loss'], axis=0) - np.std(data['train_loss'], axis=0),
                        np.mean(data['train_loss'], axis=0) + np.std(data['train_loss'], axis=0),
                        alpha=0.2, color='green')
        ax5.fill_between(range(epochs),
                        np.mean(data['test_loss'], axis=0) - np.std(data['test_loss'], axis=0),
                        np.mean(data['test_loss'], axis=0) + np.std(data['test_loss'], axis=0),
                        alpha=0.2, color='red')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.set_title('Train vs Test Loss')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Final accuracy distribution across runs
    ax6 = fig.add_subplot(gs[1, 2])
    if 'test_acc' in data and n_runs > 1:
        final_acc = data['test_acc'][:, -1]
        ax6.boxplot(final_acc, vert=True, patch_artist=True)
        ax6.set_ylabel('Final Test Accuracy (%)')
        ax6.set_title(f'Final Accuracy Distribution (n={n_runs} runs)')
        ax6.grid(True, alpha=0.3)
        # Add individual points
        for i, acc in enumerate(final_acc):
            ax6.scatter([1.1] * len(final_acc), final_acc, alpha=0.6)
    elif 'test_acc' in data:
        ax6.text(0.5, 0.5, f'Final Accuracy:\n{data["test_acc"][0, -1]:.2f}%',
                ha='center', va='center', fontsize=14, transform=ax6.transAxes)
        ax6.set_title('Single Run Result')
    ax6.set_xlim(0, 2)
    ax6.grid(True, alpha=0.3)
    
    # Save the figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n�� Plot saved to: {save_path}")
    
    plt.show()

def plot_subset_analysis(data, save_path=None):
    """Create additional analysis plots if subset data is available."""
    
    if 'subset' not in data.files:
        print("\nNo subset data available for detailed analysis.")
        return
    
    print("\nSubset data available but visualization requires additional processing.")
    print("The 'subset' array contains the indices of selected samples at each epoch.")

def main():
    """Main function to load, print, and plot CRAIG training results."""
    
    print("="*60)
    print("CRAIG Results Visualization Tool")
    print("="*60)
    
    # Load results
    data = load_latest_results('d:/tmp')
    
    if data is None:
        print("\n❌ No results found. Please run the training script first:")
        print("   cd d:\\craig-master")
        print("   python train_resnet.py -s 0.1 -w -b 64 --epochs 10 --workers 0")
        return
    
    # Print results
    print_results(data)
    
    # Create and show plots
    print("\n�� Generating plots...")
    plot_results(data, save_path='d:/tmp/craig_results_plot.png')
    
    # Additional analysis
    plot_subset_analysis(data)
    
    print("\n✅ Done! Check d:/tmp/ for the saved plot.")

if __name__ == '__main__':
    main()
