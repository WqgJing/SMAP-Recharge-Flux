import csv
import json
import os
from datetime import datetime
import time


class TrainingLogger:
    """Handles logging and statistics for PINN training with file-based logging."""

    def __init__(self, log_dir=None, experiment_name=None, hyperparams=None):
        # In-memory storage (existing functionality)
        self.losses = []
        self.comps = {
            "pde": [],
            "surf": [],
            "wt_head": [],
            "wt_kin": [],
            "ic_h": [],
            "ic_zb": [],
        }
        self.grads = {
            "pde": [],
            "surf": [],
            "wt_head": [],
            "wt_kin": [],
            "ic_h": [],
            "ic_zb": [],
            "total": [],
        }
        
        # File-based logging setup
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"pinn_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.hyperparams = hyperparams or {}
        self.start_time = time.time()
        
        # Initialize file logging if directory is provided
        if self.log_dir:
            self._setup_file_logging()
        
        # Additional metrics for hyperparameter tuning
        self.weights_history = {
            "pde": [],
            "surf": [],
            "wt_head": [],
            "wt_kin": [],
            "ic_h": [],
            "ic_zb": [],
        }
        self.cache_stats_history = {
            "mean_residual": [],
            "max_residual": [],
            "std_residual": [],
            "resample_epochs": [],
        }
        self.training_metrics = {
            "epoch_times": [],
            "learning_rates": [],
            "gradient_norms": [],
        }

    def _setup_file_logging(self):
        """Setup file logging directory and files."""
        # Create experiment directory
        self.exp_dir = os.path.join(self.log_dir, self.experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Setup CSV file for time-series metrics
        self.csv_file = os.path.join(self.exp_dir, "training_metrics.csv")
        self.csv_fieldnames = [
            "epoch", "timestamp", "elapsed_time",
            "total_loss", "pde_loss", "surf_loss", "wt_head_loss", "wt_kin_loss", "ic_h_loss", "ic_zb_loss",
            "pde_grad", "surf_grad", "wt_head_grad", "wt_kin_grad", "ic_h_grad", "ic_zb_grad", "total_grad",
            "pde_weight", "surf_weight", "wt_head_weight", "wt_kin_weight", "ic_h_weight", "ic_zb_weight",
            "cache_mean_residual", "cache_max_residual", "cache_std_residual",
            "learning_rate", "epoch_time"
        ]
        
        # Initialize CSV file with headers
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writeheader()
        
        # Save hyperparameters and experiment metadata to JSON
        self.metadata_file = os.path.join(self.exp_dir, "experiment_metadata.json")
        self._save_metadata()
        
        print(f"Logging initialized: {self.exp_dir}")

    def _save_metadata(self):
        """Save experiment metadata and hyperparameters to JSON."""
        metadata = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "hyperparameters": self.hyperparams,
            "csv_file": "training_metrics.csv",
            "description": "PINN training experiment for Richards equation with moving water table"
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def record_epoch_start(self):
        """Record start time of epoch for timing metrics."""
        self.epoch_start_time = time.time()

    def record_losses(self, total_loss, loss_dict):
        """Record loss values (existing functionality)."""
        self.losses.append(total_loss.item())
        for key in self.comps:
            self.comps[key].append(loss_dict[key].item())

    def record_gradients(self, grad_dict):
        """Record gradient norms (existing functionality)."""
        for key in grad_dict:
            self.grads[key].append(grad_dict[key])

    def record_weights(self, weights):
        """Record current weight values."""
        for key in self.weights_history:
            if key in weights:
                self.weights_history[key].append(weights[key])

    def record_cache_stats(self, cache_stats):
        """Record cache statistics."""
        if cache_stats:
            for key in self.cache_stats_history:
                if key in cache_stats and cache_stats[key]:
                    # Get the latest value
                    latest_val = cache_stats[key][-1] if isinstance(cache_stats[key], list) else cache_stats[key]
                    self.cache_stats_history[key].append(latest_val)
                else:
                    self.cache_stats_history[key].append(None)

    def record_training_metrics(self, learning_rate=None):
        """Record additional training metrics."""
        if hasattr(self, 'epoch_start_time'):
            epoch_time = time.time() - self.epoch_start_time
            self.training_metrics["epoch_times"].append(epoch_time)
        else:
            self.training_metrics["epoch_times"].append(None)
            
        self.training_metrics["learning_rates"].append(learning_rate)

    def log_epoch(self, epoch, total_loss, loss_dict, grad_dict, weights, 
                  cache_manager=None, learning_rate=None):
        """Log complete epoch information to file."""
        if not self.log_dir:
            return
            
        # Calculate timing
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        epoch_time = current_time - self.epoch_start_time if hasattr(self, 'epoch_start_time') else None
        
        # Get cache stats
        cache_mean = cache_max = cache_std = None
        if cache_manager and hasattr(cache_manager, 'cache_stats'):
            stats = cache_manager.cache_stats
            if stats["mean_residual"]:
                cache_mean = stats["mean_residual"][-1]
            if stats["max_residual"]:
                cache_max = stats["max_residual"][-1]
            if stats["std_residual"]:
                cache_std = stats["std_residual"][-1]
        
        # Prepare CSV row
        row = {
            "epoch": epoch,
            "timestamp": datetime.fromtimestamp(current_time).isoformat(),
            "elapsed_time": elapsed_time,
            "total_loss": total_loss.item(),
            "pde_loss": loss_dict["pde"].item(),
            "surf_loss": loss_dict["surf"].item(),
            "wt_head_loss": loss_dict["wt_head"].item(),
            "wt_kin_loss": loss_dict["wt_kin"].item(),
            "ic_h_loss": loss_dict["ic_h"].item(),
            "ic_zb_loss": loss_dict["ic_zb"].item(),
            "pde_grad": grad_dict["pde"],
            "surf_grad": grad_dict["surf"],
            "wt_head_grad": grad_dict["wt_head"],
            "wt_kin_grad": grad_dict["wt_kin"],
            "ic_h_grad": grad_dict["ic_h"],
            "ic_zb_grad": grad_dict["ic_zb"],
            "total_grad": grad_dict["total"],
            "pde_weight": weights["pde"],
            "surf_weight": weights["surf"],
            "wt_head_weight": weights["wt_head"],
            "wt_kin_weight": weights["wt_kin"],
            "ic_h_weight": weights["ic_h"],
            "ic_zb_weight": weights["ic_zb"],
            "cache_mean_residual": cache_mean,
            "cache_max_residual": cache_max,
            "cache_std_residual": cache_std,
            "learning_rate": learning_rate,
            "epoch_time": epoch_time
        }
        
        # Write to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writerow(row)

    def save_final_summary(self, model=None, final_metrics=None):
        """Save final training summary to JSON."""
        if not self.log_dir:
            return
            
        end_time = time.time()
        total_training_time = end_time - self.start_time
        
        summary = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(end_time).isoformat(),
            "total_training_time_seconds": total_training_time,
            "total_epochs": len(self.losses),
            "final_losses": {
                "total": self.losses[-1] if self.losses else None,
                "pde": self.comps["pde"][-1] if self.comps["pde"] else None,
                "surf": self.comps["surf"][-1] if self.comps["surf"] else None,
                "wt_head": self.comps["wt_head"][-1] if self.comps["wt_head"] else None,
                "wt_kin": self.comps["wt_kin"][-1] if self.comps["wt_kin"] else None,
                "ic_h": self.comps["ic_h"][-1] if self.comps["ic_h"] else None,
                "ic_zb": self.comps["ic_zb"][-1] if self.comps["ic_zb"] else None,
            },
            "final_gradients": {
                "total": self.grads["total"][-1] if self.grads["total"] else None,
                "pde": self.grads["pde"][-1] if self.grads["pde"] else None,
                "surf": self.grads["surf"][-1] if self.grads["surf"] else None,
                "wt_head": self.grads["wt_head"][-1] if self.grads["wt_head"] else None,
                "wt_kin": self.grads["wt_kin"][-1] if self.grads["wt_kin"] else None,
                "ic_h": self.grads["ic_h"][-1] if self.grads["ic_h"] else None,
                "ic_zb": self.grads["ic_zb"][-1] if self.grads["ic_zb"] else None,
            },
            "final_weights": {
                "pde": self.weights_history["pde"][-1] if self.weights_history["pde"] else None,
                "surf": self.weights_history["surf"][-1] if self.weights_history["surf"] else None,
                "wt_head": self.weights_history["wt_head"][-1] if self.weights_history["wt_head"] else None,
                "wt_kin": self.weights_history["wt_kin"][-1] if self.weights_history["wt_kin"] else None,
                "ic_h": self.weights_history["ic_h"][-1] if self.weights_history["ic_h"] else None,
                "ic_zb": self.weights_history["ic_zb"][-1] if self.weights_history["ic_zb"] else None,
            },
            "training_statistics": {
                "avg_epoch_time": sum(self.training_metrics["epoch_times"]) / len(self.training_metrics["epoch_times"]) if self.training_metrics["epoch_times"] else None,
                "min_total_loss": min(self.losses) if self.losses else None,
                "convergence_info": {
                    "epochs_to_min_loss": self.losses.index(min(self.losses)) + 1 if self.losses else None,
                }
            }
        }
        
        # Add any additional final metrics
        if final_metrics:
            summary["final_metrics"] = final_metrics
            
        # Save summary
        summary_file = os.path.join(self.exp_dir, "training_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Training summary saved: {summary_file}")

    def print_progress(
        self,
        epoch,
        n_epochs,
        total_loss,
        loss_dict,
        grad_dict,
        weights,
        cache_manager=None,
    ):
        """Print training progress."""
        print(f"\n[{epoch+1:4d}/{n_epochs}]")
        print(f"  Losses: total={total_loss.item():.3e}")
        print(
            f"    PDE={loss_dict['pde'].item():.3e}, "
            f"Surf={loss_dict['surf'].item():.3e}"
        )
        print(
            f"    WT(h)={loss_dict['wt_head'].item():.3e}, "
            f"WT(kin)={loss_dict['wt_kin'].item():.3e}"
        )
        print(
            f"    IC(h)={loss_dict['ic_h'].item():.3e}, "
            f"IC(zb)={loss_dict['ic_zb'].item():.3e}"
        )

        print(f"  Weighted Gradients (L2 norm):")
        print(f"    PDE={grad_dict['pde']:.3e}, Surf={grad_dict['surf']:.3e}")
        print(
            f"    WT(h)={grad_dict['wt_head']:.3e}, "
            f"WT(kin)={grad_dict['wt_kin']:.3e}"
        )
        print(f"    IC(h)={grad_dict['ic_h']:.3e}, IC(zb)={grad_dict['ic_zb']:.3e}")
        print(f"    Total={grad_dict['total']:.3e}")

        print(f"  Current weights:")
        for key in weights:
            print(f"    {key}: {weights[key]:.3e}")

        # Print cache statistics if available
        if cache_manager is not None:
            cache_manager.print_stats(epoch)

    def print_final_summary(self, total_grad_norm, weights, cache_manager=None):
        """Print final training summary."""
        print("\nTraining done.")
        print(f"Final gradient norms: Total={total_grad_norm:.3e}")
        print(f"Final weights:")
        for key in weights:
            print(f"  {key}: {weights[key]:.3e}")

        if cache_manager is not None:
            print(f"\nCache statistics:")
            print(
                f"  Total resamples: {len(cache_manager.cache_stats['resample_epochs'])}"
            )
            if cache_manager.cache_stats["mean_residual"]:
                print(
                    f"  Final mean residual: "
                    f"{cache_manager.cache_stats['mean_residual'][-1]:.3e}"
                )
                print(
                    f"  Final max residual: "
                    f"{cache_manager.cache_stats['max_residual'][-1]:.3e}"
                )
