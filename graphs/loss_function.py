import argparse
import wandb
import matplotlib.pyplot as plt
import pandas as pd

def plot_loss_curves(entity: str, project: str, run_id: str):
    """
    Fetches and plots train/loss and eval/loss curves from a given wandb run id.

    Args:
        entity (str): The wandb entity (username or team name).
        project (str): The wandb project name.
        run_id (str): The ID of the wandb run.
    """
    # Initialize the wandb API
    api = wandb.Api()
    run_path = f"{entity}/{project}/{run_id}"

    try:
        # Get the specified run
        run = api.run(run_path)
    except wandb.errors.CommError as e:
        print(f"Failed to get run '{run_path}': {e}")
        return

    # Get history, scanning only for the keys we need
    history = run.history(keys=["train/loss", "eval/loss", "_step"])

    if history.empty:
        print(f"No history data found for run '{run_path}'.")
        return

    # Extract train/loss and eval/loss respectively, and remove NaN values
    train_loss_data = history[["_step", "train/loss"]].dropna()
    eval_loss_data = history[["_step", "eval/loss"]].dropna()

    # Create the plot
    plt.figure(figsize=(12, 7))

    if not train_loss_data.empty:
        plt.plot(train_loss_data["_step"], train_loss_data["train/loss"], label="train/loss")
    else:
        print("No 'train/loss' data found.")

    if not eval_loss_data.empty:
        plt.plot(eval_loss_data["_step"], eval_loss_data["eval/loss"], label="eval/loss")
    else:
        print("No 'eval/loss' data found.")

    # Set chart title and axis labels
    plt.title(f"Loss Curves for Run '{run_id}'")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Save the chart to a file
    output_filename = f"loss_curve_{run_id}.png"
    plt.savefig(output_filename)
    print(f"Chart saved to: {output_filename}")

    # Display the chart
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and plot train/eval loss curves from a wandb run.")
    parser.add_argument("--entity", type=str, default="pufanyi", help="Wandb entity (username or team name).")
    parser.add_argument("--project", type=str, default="sc4001", help="Wandb project name.")
    parser.add_argument("--run_id", type=str, required=True, help="Wandb run ID.")

    args = parser.parse_args()

    plot_loss_curves(args.entity, args.project, args.run_id)
