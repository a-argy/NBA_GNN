"""
Visualization functions for GAT shot prediction model.
Plots training history, loss curves, accuracy metrics, and evaluation metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, confusion_matrix
import seaborn as sns


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot comprehensive training history with multiple subplots.
    
    Args:
        history (dict): Dictionary containing training metrics with keys:
            - 'train_loss': List of training losses per epoch
            - 'train_acc': List of training accuracies per epoch
            - 'val_loss': List of validation losses per epoch
            - 'val_acc': List of validation accuracies per epoch
            - 'val_acc_0': List of validation accuracy for class 0 (missed)
            - 'val_acc_1': List of validation accuracy for class 1 (made)
        save_path (str): Path to save the figure
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training and Validation Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training and Validation Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Per-Class Validation Accuracy
    axes[1, 0].plot(epochs, history['val_acc_0'], 'g-', label='Missed Shots (Class 0)', linewidth=2)
    axes[1, 0].plot(epochs, history['val_acc_1'], 'm-', label='Made Shots (Class 1)', linewidth=2)
    axes[1, 0].plot(epochs, history['val_acc'], 'r--', label='Overall Accuracy', linewidth=2, alpha=0.7)
    axes[1, 0].set_title('Validation Accuracy by Class', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy', fontsize=12)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Overfitting Indicator (Loss Difference)
    loss_diff = [v - t for v, t in zip(history['val_loss'], history['train_loss'])]
    axes[1, 1].plot(epochs, loss_diff, 'orange', label='Val Loss - Train Loss', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_title('Overfitting Indicator (Loss Difference)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Validation Loss - Training Loss', fontsize=12)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Fill areas for overfitting/underfitting
    axes[1, 1].fill_between(epochs, 0, loss_diff, 
                            where=[d > 0 for d in loss_diff], 
                            alpha=0.3, color='red', label='Overfitting')
    axes[1, 1].fill_between(epochs, 0, loss_diff, 
                            where=[d <= 0 for d in loss_diff], 
                            alpha=0.3, color='green', label='Underfitting')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining history plot saved as '{save_path}'")
    plt.show()


def plot_loss_curves(history, save_path='loss_curves.png'):
    """
    Plot simple loss curves for train and validation.
    
    Args:
        history (dict): Dictionary with 'train_loss' and 'val_loss' keys
        save_path (str): Path to save the figure
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=4)
    plt.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Loss curves saved as '{save_path}'")
    plt.show()


def plot_accuracy_curves(history, save_path='accuracy_curves.png'):
    """
    Plot accuracy curves for train and validation.
    
    Args:
        history (dict): Dictionary with 'train_acc' and 'val_acc' keys
        save_path (str): Path to save the figure
    """
    epochs = range(1, len(history['train_acc']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2, markersize=4)
    plt.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=4)
    plt.title('Training and Validation Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Accuracy curves saved as '{save_path}'")
    plt.show()


def plot_class_accuracy(history, save_path='class_accuracy.png'):
    """
    Plot per-class accuracy over training.
    
    Args:
        history (dict): Dictionary with 'val_acc_0', 'val_acc_1', 'val_acc' keys
        save_path (str): Path to save the figure
    """
    epochs = range(1, len(history['val_acc']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_acc_0'], 'g-o', label='Missed Shots (Class 0)', linewidth=2, markersize=4)
    plt.plot(epochs, history['val_acc_1'], 'm-s', label='Made Shots (Class 1)', linewidth=2, markersize=4)
    plt.plot(epochs, history['val_acc'], 'r--', label='Overall Accuracy', linewidth=2, alpha=0.7)
    plt.title('Validation Accuracy by Class', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Class accuracy plot saved as '{save_path}'")
    plt.show()


def plot_roc_curve(results, save_path='roc_curve.png'):
    """
    Plot ROC curve for test set predictions.
    
    Args:
        results (dict): Dictionary containing 'test_probs' and 'test_labels'
        save_path (str): Path to save the figure
    """
    if 'test_probs' not in results or 'test_labels' not in results:
        print("Warning: ROC curve requires 'test_probs' and 'test_labels' in results")
        return
    
    y_true = results['test_labels']
    y_probs = results['test_probs']
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"ROC curve saved as '{save_path}'")
    plt.show()


def plot_confusion_matrix(results, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix for test set predictions.
    
    Args:
        results (dict): Dictionary containing 'test_preds' and 'test_labels'
        save_path (str): Path to save the figure
    """
    if 'test_preds' not in results or 'test_labels' not in results:
        print("Warning: Confusion matrix requires 'test_preds' and 'test_labels' in results")
        return
    
    y_true = results['test_labels']
    y_pred = results['test_preds']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Missed', 'Made'],
                yticklabels=['Missed', 'Made'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved as '{save_path}'")
    plt.show()


def plot_metrics_bar(results, save_path='metrics_comparison.png'):
    """
    Plot bar chart comparing different evaluation metrics.
    
    Args:
        results (dict): Dictionary containing metric values
        save_path (str): Path to save the figure
    """
    metrics = {
        'Accuracy': results.get('test_acc', 0),
        'ROC AUC': results.get('test_roc_auc', 0),
        'F1 Score': results.get('test_f1', 0),
        'Precision': results.get('test_precision', 0),
        'Recall': results.get('test_recall', 0)
    }
    
    # Filter out metrics with 0 values (not computed)
    metrics = {k: v for k, v in metrics.items() if v > 0}
    
    if not metrics:
        print("Warning: No metrics available to plot")
        return
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics.keys(), metrics.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    plt.ylim([0, 1.0])
    plt.ylabel('Score', fontsize=14)
    plt.title('Test Set Evaluation Metrics', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Metrics bar chart saved as '{save_path}'")
    plt.show()


def print_summary(results):
    """
    Print a comprehensive summary of training results with evaluation metrics.
    
    Args:
        results (dict): Dictionary containing training results
    """
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    # Best validation accuracy (handle missing key)
    if 'best_val_acc' in results and 'best_epoch' in results:
        print(f"\nBest Validation Accuracy: {results['best_val_acc']:.4f} (epoch {results['best_epoch']})")
    
    # Test set performance
    print(f"\nTest Set Performance:")
    if 'test_acc' in results:
        print(f"  Overall Accuracy:      {results['test_acc']:.4f}")
    if 'test_acc_0' in results:
        print(f"  Missed Shots (Class 0): {results['test_acc_0']:.4f}")
    if 'test_acc_1' in results:
        print(f"  Made Shots (Class 1):   {results['test_acc_1']:.4f}")
    if 'test_loss' in results:
        print(f"  Test Loss:             {results['test_loss']:.4f}")
    
    # Print additional metrics if available
    if 'test_roc_auc' in results:
        print(f"\nClassification Metrics:")
        print(f"  ROC AUC:      {results['test_roc_auc']:.4f}")
        print(f"  F1 Score:     {results['test_f1']:.4f}")
        print(f"  Precision:    {results['test_precision']:.4f}")
        print(f"  Recall:       {results['test_recall']:.4f}")
    
    # Baseline comparison
    if 'baseline_acc' in results and 'test_acc' in results:
        print(f"\nBaseline (majority class): {results['baseline_acc']:.4f}")
        print(f"Improvement over baseline: {(results['test_acc'] - results['baseline_acc'])*100:.2f}%")
    
    # Model save path
    if 'model_save_path' in results:
        print(f"\nModel saved to: {results['model_save_path']}")
    
    print("="*60)


def visualize_all(results, output_dir='.'):
    """
    Generate all visualization plots.
    
    Args:
        results (dict): Dictionary containing training results with 'history' key
        output_dir (str): Directory to save plots
    """
    import os
    
    history = results['history']
    
    # Generate all plots
    plot_training_history(history, os.path.join(output_dir, 'training_history.png'))
    plot_loss_curves(history, os.path.join(output_dir, 'loss_curves.png'))
    plot_accuracy_curves(history, os.path.join(output_dir, 'accuracy_curves.png'))
    plot_class_accuracy(history, os.path.join(output_dir, 'class_accuracy.png'))
    
    # Generate evaluation metric plots if available
    if 'test_probs' in results and 'test_labels' in results:
        plot_roc_curve(results, os.path.join(output_dir, 'roc_curve.png'))
    
    if 'test_preds' in results and 'test_labels' in results:
        plot_confusion_matrix(results, os.path.join(output_dir, 'confusion_matrix.png'))
    
    if 'test_roc_auc' in results:
        plot_metrics_bar(results, os.path.join(output_dir, 'metrics_comparison.png'))
    
    # Print summary
    print_summary(results)


# Example usage
if __name__ == "__main__":
    print("This module provides visualization functions.")
    print("Import and use after training:")
    print()
    print("  from visualize import plot_training_history, print_summary")
    print("  from data import load_graph_data")
    print("  from train import train_model")
    print()
    print("  graphs, labels = load_graph_data()")
    print("  model, results = train_model(graphs, labels)")
    print("  plot_training_history(results['history'])")
    print("  print_summary(results)")
