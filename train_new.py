import argparse
import os
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.cut_gan import MultiStageGAN
from data.dataset_oral import XYDataset
from utils.util import read_yaml_config, transforms, reverse_image_normalize

def save_images(images_dict, save_dir, epoch, iteration):
    """Helper function to save images with proper naming."""
    for img_name, img in images_dict.items():
        save_path = os.path.join(save_dir, f"epoch_{epoch}_iter_{iteration}_{img_name}.png")
        save_image(reverse_image_normalize(img), save_path)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Multi-Stage GAN UI2I framework")
    parser.add_argument("-c", "--config", default="./config.yaml", help="Path to the yaml config file")
    parser.add_argument("-v", "--verbose", help="Verbose mode", action="store_true")
    args = parser.parse_args()
    
    # Load configuration
    config = read_yaml_config(args.config)
    
    # Create experiment directories
    exp_dir = os.path.join(config["EXPERIMENT_ROOT_PATH"], config["EXPERIMENT_NAME"])
    train_dir = os.path.join(exp_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    
    # Initialize model
    model = MultiStageGAN(config)

    # Initialize dataset and dataloader
    dataset = XYDataset(
        root_A=config["TRAINING_SETTING"]["TRAIN_DIR_A"],
        root_A_mask=config["TRAINING_SETTING"]["TRAIN_DIR_A_MASK"],
        root_B=config["TRAINING_SETTING"]["TRAIN_DIR_B"],
        root_B_mask=config["TRAINING_SETTING"]["TRAIN_DIR_B_MASK"],
        transform=transforms
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["TRAINING_SETTING"]["BATCH_SIZE"],
        shuffle=True,
        num_workers=config["TRAINING_SETTING"]["NUM_WORKERS"],
        pin_memory=True
    )

    # Training loop
    for epoch in range(config["TRAINING_SETTING"]["NUM_EPOCHS"]):
        # Initialize loss tracking
        epoch_losses = defaultdict(float)
        running_losses = defaultdict(float)
        
        for idx, data in enumerate(dataloader):
            if args.verbose:
                print(f"[Epoch {epoch}][Iter {idx}] Processing ...", end="\r")
            
            # Train step
            model.set_input(data)
            model.optimize_parameters()
            
            # Accumulate losses
            current_losses = model.get_current_losses()
            for k, v in current_losses.items():
                epoch_losses[k] += v
                running_losses[k] += v

            # Visualization and logging
            if idx % config["TRAINING_SETTING"]["VISUALIZATION_STEP"] == 0 and idx > 0:
                # Get and save current visual results
                visuals = model.get_current_visuals()
                save_images(visuals, train_dir, epoch, idx)
                
                # Calculate and log running averages
                steps = config["TRAINING_SETTING"]["VISUALIZATION_STEP"]
                avg_losses = {k: v/steps for k, v in running_losses.items()}
                
                # Print progress
                loss_str = " ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()])
                print(f"[Epoch {epoch}][Iter {idx}] {loss_str}", flush=True)
                
                # Reset running losses
                running_losses = defaultdict(float)

        # Epoch end processing
        model.scheduler_step()
        
        # Calculate and log epoch averages
        num_iterations = len(dataloader)
        epoch_avg_losses = {k: v/num_iterations for k, v in epoch_losses.items()}
        epoch_loss_str = " ".join([f"{k}: {v:.4f}" for k, v in epoch_avg_losses.items()])
        print(f"[Epoch {epoch}] Average losses: {epoch_loss_str}", flush=True)
        
        # Save model checkpoints
        if epoch % config["TRAINING_SETTING"]["SAVE_MODEL_EPOCH_STEP"] == 0 and \
           config["TRAINING_SETTING"]["SAVE_MODEL"]:
            model.save_networks(epoch)
            
            # Save latest visuals as reference
            latest_visuals = model.get_current_visuals()
            save_images(latest_visuals, train_dir, epoch, "final")
    
    # Save final model
    model.save_networks("latest")

if __name__ == "__main__":
    main()