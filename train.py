import argparse
from src.training import *
from src.utils import load_and_prepare_data
import torch
import os
import time

torch.backends.cudnn.benchmark = True
def main():
    parser = argparse.ArgumentParser(description="Train a 3-model pipeline")
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to load')
    parser.add_argument('--mode', type=str, default='baseline', help='Dataset loading mode')
    parser.add_argument('--epochs', type=int, default=1200, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')
    parser.add_argument('--experiment', type=int, default=8, help='Number of experiments')
    args = parser.parse_args()

    train_loader, val_loader, test_loader, num_features = load_and_prepare_data(args.dataset, args.mode, args.batch_size, args.num_workers)
    model = create_model(num_features, train_loader.dataset.x.shape[0], args.mode)

    # start_training = time.time()
    run_training(model, train_loader, val_loader, args.mode, args.dataset, args.epochs)
    # end_training = time.time()

    checkpoint_dir = f'./model_checkpoints/{args.dataset}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    torch.save(model, f'{checkpoint_dir}/{args.mode}.pth')

    if not os.path.exists(f'./results/{args.dataset}/{args.experiment}/'):
        os.makedirs(f'./results/{args.dataset}/{args.experiment}/')

    # start_testing = time.time()
    run_test_and_save_results(
        model,
        test_loader=test_loader,
        output_path=f"./results/{args.dataset}/{args.experiment}/{args.dataset}_{args.mode}_test_results.txt",
    )
    # end_testing = time.time()

    # with open(f'{args.dataset}_{args.mode}_training_time.txt', 'w') as f:
    #     f.write(f"Training time: {end_training - start_training} seconds\n Testing time: {end_testing - start_testing} seconds\n")

if __name__ == '__main__':
    main()
