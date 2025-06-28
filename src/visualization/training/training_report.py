"""Training report generation for comprehensive training analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class TrainingReportGenerator:
    """Generates comprehensive training reports with visualizations."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize training report generator.
        
        Args:
            save_dir: Directory to save reports
        """
        self.save_dir = save_dir or Path("data/figures/reports")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def generate_training_summary(self, 
                                training_history: Dict[str, List[float]],
                                model_config: Dict[str, Any],
                                best_metrics: Dict[str, float],
                                save_name: str = "training_summary") -> None:
        """
        Generate comprehensive training summary report.
        
        Args:
            training_history: Dictionary with training metrics over epochs
            model_config: Model configuration details
            best_metrics: Best achieved metrics
            save_name: Base filename for report
        """
        try:
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(20, 16))
            
            # Extract data
            epochs = list(range(len(training_history.get('train_loss', []))))
            train_loss = training_history.get('train_loss', [])
            val_loss = training_history.get('val_loss', [])
            train_acc = training_history.get('train_accuracy', [])
            val_acc = training_history.get('val_accuracy', [])
            
            # Plot 1: Loss curves
            ax1 = plt.subplot(3, 3, 1)
            if train_loss:
                plt.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
            if val_loss:
                plt.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Accuracy curves
            ax2 = plt.subplot(3, 3, 2)
            if train_acc:
                plt.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
            if val_acc:
                plt.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training and Validation Accuracy', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Learning rate schedule
            ax3 = plt.subplot(3, 3, 3)
            lr_history = training_history.get('learning_rate', [])
            if lr_history:
                plt.plot(epochs, lr_history, 'g-', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.title('Learning Rate Schedule', fontweight='bold')
                plt.yscale('log')
                plt.grid(True, alpha=0.3)
            
            # Plot 4: Loss distribution
            ax4 = plt.subplot(3, 3, 4)
            if train_loss and val_loss:
                plt.hist(train_loss, bins=20, alpha=0.7, label='Train Loss', density=True)
                plt.hist(val_loss, bins=20, alpha=0.7, label='Val Loss', density=True)
                plt.xlabel('Loss Value')
                plt.ylabel('Density')
                plt.title('Loss Distribution', fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Plot 5: Overfitting analysis
            ax5 = plt.subplot(3, 3, 5)
            if train_loss and val_loss:
                gap = np.array(val_loss) - np.array(train_loss)
                plt.plot(epochs, gap, 'purple', linewidth=2)
                plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                plt.xlabel('Epoch')
                plt.ylabel('Validation - Training Loss')
                plt.title('Overfitting Analysis', fontweight='bold')
                plt.grid(True, alpha=0.3)
            
            # Plot 6: Best metrics bar chart
            ax6 = plt.subplot(3, 3, 6)
            if best_metrics:
                metric_names = list(best_metrics.keys())
                metric_values = list(best_metrics.values())
                bars = plt.bar(range(len(metric_names)), metric_values, 
                             color=sns.color_palette("viridis", len(metric_names)))
                plt.xticks(range(len(metric_names)), metric_names, rotation=45)
                plt.ylabel('Value')
                plt.title('Best Metrics Achieved', fontweight='bold')
                
                # Add value labels on bars
                for bar, value in zip(bars, metric_values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 7: Training stability
            ax7 = plt.subplot(3, 3, 7)
            if val_loss:
                # Calculate moving average and std
                window = min(10, len(val_loss) // 4)
                if window > 1:
                    moving_avg = pd.Series(val_loss).rolling(window).mean()
                    moving_std = pd.Series(val_loss).rolling(window).std()
                    plt.plot(epochs, val_loss, alpha=0.3, color='blue')
                    plt.plot(epochs, moving_avg, 'b-', linewidth=2, label='Moving Avg')
                    plt.fill_between(epochs, moving_avg - moving_std, 
                                   moving_avg + moving_std, alpha=0.3, label='±1σ')
                    plt.xlabel('Epoch')
                    plt.ylabel('Validation Loss')
                    plt.title('Training Stability', fontweight='bold')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
            
            # Configuration table
            ax8 = plt.subplot(3, 1, 3)
            ax8.axis('tight')
            ax8.axis('off')
            
            # Create configuration table
            config_data = []
            for key, value in model_config.items():
                config_data.append([key, str(value)])
            
            # Add training summary
            summary_data = [
                ['Total Epochs', str(len(epochs))],
                ['Best Val Loss', f"{min(val_loss):.4f}" if val_loss else "N/A"],
                ['Best Val Acc', f"{max(val_acc):.4f}" if val_acc else "N/A"],
                ['Final Train Loss', f"{train_loss[-1]:.4f}" if train_loss else "N/A"],
                ['Final Val Loss', f"{val_loss[-1]:.4f}" if val_loss else "N/A"]
            ]
            
            all_data = config_data + [['', '']] + summary_data
            
            table = ax8.table(cellText=all_data,
                            colLabels=['Configuration', 'Value'],
                            cellLoc='left',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            plt.suptitle(f'Training Summary Report - {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                        fontsize=18, fontweight='bold')
            plt.tight_layout()
            
            plt.savefig(self.save_dir / f"{save_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training summary report saved to {save_name}.png")
            
        except Exception as e:
            logger.error(f"Error generating training summary: {e}")
            raise
    
    def generate_convergence_analysis(self, 
                                    training_history: Dict[str, List[float]],
                                    save_name: str = "convergence_analysis") -> Dict[str, Any]:
        """
        Analyze and visualize training convergence.
        
        Args:
            training_history: Dictionary with training metrics
            save_name: Filename to save plot
            
        Returns:
            Dictionary with convergence statistics
        """
        try:
            train_loss = training_history.get('train_loss', [])
            val_loss = training_history.get('val_loss', [])
            
            if not train_loss:
                logger.warning("No training loss data for convergence analysis")
                return {}
            
            epochs = list(range(len(train_loss)))
            
            # Calculate convergence metrics
            convergence_stats = {}
            
            # Smoothness: variance of differences
            if len(train_loss) > 1:
                train_diffs = np.diff(train_loss)
                convergence_stats['train_smoothness'] = 1 / (1 + np.var(train_diffs))
                
                if val_loss and len(val_loss) > 1:
                    val_diffs = np.diff(val_loss)
                    convergence_stats['val_smoothness'] = 1 / (1 + np.var(val_diffs))
            
            # Convergence point: when loss stops decreasing significantly
            threshold = 0.001  # 0.1% improvement threshold
            convergence_epoch = len(train_loss)
            
            for i in range(10, len(train_loss)):
                recent_improvement = (train_loss[i-10] - train_loss[i]) / train_loss[i-10]
                if recent_improvement < threshold:
                    convergence_epoch = i
                    break
            
            convergence_stats['convergence_epoch'] = convergence_epoch
            convergence_stats['converged'] = convergence_epoch < len(train_loss)
            
            # Learning efficiency: final improvement / epochs
            if len(train_loss) > 1:
                total_improvement = (train_loss[0] - train_loss[-1]) / train_loss[0]
                convergence_stats['learning_efficiency'] = total_improvement / len(train_loss)
            
            # Create visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Loss curves with convergence point
            ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
            if val_loss:
                ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
            
            if convergence_stats.get('converged', False):
                ax1.axvline(x=convergence_epoch, color='green', linestyle='--', 
                           label=f'Convergence (Epoch {convergence_epoch})')
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Loss Convergence Analysis')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Loss differences (derivatives)
            if len(train_loss) > 1:
                ax2.plot(epochs[1:], train_diffs, 'b-', alpha=0.7, label='Train Loss Δ')
                if val_loss and len(val_loss) > 1:
                    ax2.plot(epochs[1:], val_diffs, 'r-', alpha=0.7, label='Val Loss Δ')
                
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.axhline(y=-threshold * np.mean(train_loss), color='green', 
                           linestyle='--', label='Convergence Threshold')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss Change')
                ax2.set_title('Loss Gradient Analysis')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Plot 3: Moving average convergence
            window = min(10, len(train_loss) // 4)
            if window > 1:
                train_ma = pd.Series(train_loss).rolling(window).mean()
                ax3.plot(epochs, train_loss, alpha=0.3, color='blue')
                ax3.plot(epochs, train_ma, 'b-', linewidth=3, label=f'MA({window})')
                
                if val_loss:
                    val_ma = pd.Series(val_loss).rolling(window).mean()
                    ax3.plot(epochs, val_loss, alpha=0.3, color='red')
                    ax3.plot(epochs, val_ma, 'r-', linewidth=3, label=f'Val MA({window})')
                
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Loss')
                ax3.set_title('Smoothed Convergence')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Plot 4: Convergence statistics
            ax4.axis('off')
            stats_text = []
            for key, value in convergence_stats.items():
                if isinstance(value, float):
                    stats_text.append(f"{key}: {value:.4f}")
                else:
                    stats_text.append(f"{key}: {value}")
            
            ax4.text(0.1, 0.9, "Convergence Statistics:", 
                    fontsize=14, fontweight='bold', transform=ax4.transAxes)
            
            for i, text in enumerate(stats_text):
                ax4.text(0.1, 0.8 - i*0.1, text, 
                        fontsize=12, transform=ax4.transAxes)
            
            plt.suptitle('Training Convergence Analysis', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plt.savefig(self.save_dir / f"{save_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            return convergence_stats
            
        except Exception as e:
            logger.error(f"Error in convergence analysis: {e}")
            raise
    
    def generate_comparison_report(self, 
                                 experiments: Dict[str, Dict[str, List[float]]],
                                 save_name: str = "experiments_comparison") -> None:
        """
        Compare multiple training experiments.
        
        Args:
            experiments: Dict mapping experiment names to training histories
            save_name: Filename to save plot
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(experiments)))
            
            # Plot 1: Training loss comparison
            for i, (exp_name, history) in enumerate(experiments.items()):
                train_loss = history.get('train_loss', [])
                if train_loss:
                    epochs = list(range(len(train_loss)))
                    ax1.plot(epochs, train_loss, color=colors[i], 
                           label=exp_name, linewidth=2)
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Training Loss')
            ax1.set_title('Training Loss Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Validation loss comparison
            for i, (exp_name, history) in enumerate(experiments.items()):
                val_loss = history.get('val_loss', [])
                if val_loss:
                    epochs = list(range(len(val_loss)))
                    ax2.plot(epochs, val_loss, color=colors[i], 
                           label=exp_name, linewidth=2)
            
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Validation Loss')
            ax2.set_title('Validation Loss Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Final metrics comparison
            exp_names = list(experiments.keys())
            final_train_loss = []
            final_val_loss = []
            best_val_acc = []
            
            for exp_name, history in experiments.items():
                train_loss = history.get('train_loss', [])
                val_loss = history.get('val_loss', [])
                val_acc = history.get('val_accuracy', [])
                
                final_train_loss.append(train_loss[-1] if train_loss else 0)
                final_val_loss.append(val_loss[-1] if val_loss else 0)
                best_val_acc.append(max(val_acc) if val_acc else 0)
            
            x_pos = np.arange(len(exp_names))
            width = 0.25
            
            ax3.bar(x_pos - width, final_train_loss, width, 
                   label='Final Train Loss', alpha=0.7)
            ax3.bar(x_pos, final_val_loss, width, 
                   label='Final Val Loss', alpha=0.7)
            ax3.bar(x_pos + width, best_val_acc, width, 
                   label='Best Val Acc', alpha=0.7)
            
            ax3.set_xlabel('Experiment')
            ax3.set_ylabel('Value')
            ax3.set_title('Final Metrics Comparison')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(exp_names, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Convergence speed comparison
            convergence_epochs = []
            for exp_name, history in experiments.items():
                train_loss = history.get('train_loss', [])
                if len(train_loss) > 10:
                    # Find convergence point
                    threshold = 0.001
                    convergence_epoch = len(train_loss)
                    
                    for i in range(10, len(train_loss)):
                        recent_improvement = (train_loss[i-10] - train_loss[i]) / train_loss[i-10]
                        if recent_improvement < threshold:
                            convergence_epoch = i
                            break
                    
                    convergence_epochs.append(convergence_epoch)
                else:
                    convergence_epochs.append(len(train_loss))
            
            bars = ax4.bar(range(len(exp_names)), convergence_epochs, 
                          color=colors, alpha=0.7)
            ax4.set_xlabel('Experiment')
            ax4.set_ylabel('Convergence Epoch')
            ax4.set_title('Convergence Speed Comparison')
            ax4.set_xticks(range(len(exp_names)))
            ax4.set_xticklabels(exp_names, rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, epoch in zip(bars, convergence_epochs):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(epoch), ha='center', va='bottom', fontweight='bold')
            
            plt.suptitle('Experiments Comparison Report', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plt.savefig(self.save_dir / f"{save_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Experiments comparison report saved to {save_name}.png")
            
        except Exception as e:
            logger.error(f"Error generating comparison report: {e}")
            raise 