#!/usr/bin/env python3
import argparse
import sys
import torch
import torch.nn
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
import torchvision.transforms
import torchvision.datasets
import glob
#from pathlib import Path
#sys.path.append(str(Path('.').absolute().parent))

from BT2 import *
from cross_entropy import LabelSmoothingCrossEntropy
import writeLogAcc as wA

def get_args():
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description='TickNet training script for CIFAR and fine-grained datasets.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('-r', '--data-root', type=str, required=True, help='Dataset root path.')
    parser.add_argument('-r', '--data-root', type=str, default='../data', help='Dataset root path.')
    #parser.add_argument('-d', '--dataset', choices=['cifar10', 'cifar100', 'dogs'], required=True, help='Dataset name.')
    parser.add_argument('-d', '--dataset', type=str, choices=['cifar10', 'cifar100', 'dogs'], default='cifar10', help='Dataset name.')
    parser.add_argument('--download', action='store_true', help='Download the specified dataset before running the training.')
    #parser.add_argument('-a', '--architecture', type=str, required=True, help='Model architecture name.')
    parser.add_argument('-a', '--architecture', type=str, default='mobilenetv1_w1', help='Model architecture name.')
    parser.add_argument('-g', '--gpu-id', default=1, type=int, help='ID of the GPU to use. Set to -1 to use CPU.')
    parser.add_argument('-j', '--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('-b', '--batch-size', default=16, type=int, help='Batch size.')
    parser.add_argument('-e', '--epochs', default=200, type=int, help='Number of total epochs to run.')
    parser.add_argument('-l', '--learning-rate', default=0.1, type=float, help='Initial learning rate.')
    parser.add_argument('-s', '--schedule', nargs='+', default=[100, 150, 180], type=int, help='Learning rate schedule (epochs after which the learning rate should be dropped).')    
    parser.add_argument('-m', '--momentum', default=0.9, type=float, help='SGD momentum.')
    parser.add_argument('-w', '--weight-decay', default=1e-4, type=float, help='SGD weight decay.')
    #parser.add_argument('--alpha', default=0.1, type=float, help='BSConv-S weighting coefficient for the regularization loss.')
    parser.add_argument('-re','--resume', type=str, default='', help='Path to checkpoint to resume training from')
    parser.add_argument('-se','--start-epoch', type=int, default=0,
                        help='Starting epoch number (useful for correct naming/logging when resuming)')
    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the specified directory."""
    if not os.path.exists(checkpoint_dir):
        return None

    # Find all checkpoint files
    checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch*.pth')
    checkpoint_files = glob.glob(checkpoint_pattern)

    if not checkpoint_files:
        return None

    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.split('epoch')[-1].split('_')[0]))
    return checkpoint_files[-1]


def load_checkpoint(args, model, optimizer, scheduler):
    """Load the latest checkpoint or a specific checkpoint file."""
    checkpoint_path = args.resume

    # If a directory is specified, find the latest checkpoint
    if os.path.isdir(checkpoint_path):
        checkpoint_path = find_latest_checkpoint(checkpoint_path)
        if checkpoint_path is None:
            print(f"No checkpoints found in directory '{args.resume}'")
            return None, None, 0

    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at '{checkpoint_path}'")
        return None, None, 0

    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state if it exists

    # Load scheduler state if it exists
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Get epoch number from checkpoint filename or stored epoch
    if args.start_epoch is None:
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        else:
            # Extract epoch from filename
            epoch_str = checkpoint_path.split('epoch')[-1].split('_')[0]
            start_epoch = int(epoch_str)
    else:
        start_epoch = args.start_epoch

    # Get best accuracy information
    acc_val_max = checkpoint.get('acc_val_max', None)
    acc_val_argmax = checkpoint.get('acc_val_argmax', None)

    print(f"Loaded checkpoint (epoch {start_epoch})")
    if acc_val_max is not None:
        print(f"Best accuracy so far: {acc_val_max * 100:.2f}% at epoch {acc_val_argmax + 1}")

    return acc_val_max, acc_val_argmax, start_epoch

def get_device(args):
    """
    Determine the device to use for the given arguments.
    """
    if args.gpu_id >= 0:
        return torch.device('cuda:{}'.format(args.gpu_id))
    else:
        return torch.device('cpu')
    

def get_data_loader(args, train):
    """
    Return the data loader for the given arguments.
    """
    if args.dataset in ('cifar10', 'cifar100'):
        # select transforms based on train/val
        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
    
        # cifar10 vs. cifar100
        if args.dataset == 'cifar10':
            dataset_class = torchvision.datasets.CIFAR10
        else:
            dataset_class = torchvision.datasets.CIFAR100
            
    elif args.dataset in ('dogs',):
        # select transforms based on train/val
        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(256, 256)),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(0.4),
                torchvision.transforms.ToTensor()
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(256, 256)),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor()
            ])
    
        # dataset_class = bsconv.datasets.StanfordDogs
    
    else:
        raise NotImplementedError('Can\'t determine data loader for dataset \'{}\''.format(args.dataset))
    
    # trigger download only once
    if args.download:
        dataset_class(root=args.data_root, train=train, download=True, transform=transform)

    # instantiate dataset class and create data loader from it
    dataset = dataset_class(root=args.data_root, train=train, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True if train else False, num_workers=args.workers)    


def calculate_accuracy(output, target):
    """
    Top-1 classification accuracy.
    """
    with torch.no_grad():
        batch_size = output.shape[0]
        prediction = torch.argmax(output, dim=1)
        return torch.sum(prediction == target).item() / batch_size


def run_epoch(train, data_loader, model, criterion, optimizer, n_epoch, args, device):
    """
    Run one epoch. If `train` is `True` perform training, otherwise validate.
    """
    if train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)
    
    batch_count = len(data_loader)
    losses = []
    accs = []
    for (n_batch, (images, target)) in enumerate(data_loader):
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = criterion(output, target)
        
        # record loss and measure accuracy
        loss_item = loss.item()
        losses.append(loss_item)
        acc = calculate_accuracy(output, target)
        accs.append(acc)

        # compute gradient and do SGD step
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        if (n_batch % 40) == 0:
            print('[{}]  epoch {}/{},  batch {}/{},  loss_{}={:.5f},  acc_{}={:.2f}%'.format('train' if train else ' val ', n_epoch + 1, args.epochs, n_batch + 1, batch_count, "train" if train else "val", loss_item, "train" if train else "val", 100.0 * acc))
    
    return (sum(losses) / len(losses), sum(accs) / len(accs))
            

def main():
    """
    Run the complete model training.
    """
    args = get_args()
    print('Command: {}'.format(' '.join(sys.argv)))
    args.gpu_id = 0
    device = get_device(args)
    print('Using device {}'.format(device))
    
    # print model with parameter and FLOPs counts    
    torch.autograd.set_detect_anomaly(True)     
    
    
      
    strmode = 'My_Net_'  
    pathout = './checkpoints/CIFAR10_My_Net/' + strmode
    filenameLOG = pathout + '/' + strmode + '.txt'
    if not os.path.exists(pathout):
        os.makedirs(pathout)
    # get model
     
    model = build_Net(10, cifar=True)
    model = model.to(device)
    
    print(model)
    
    print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))
    
    # define loss function and optimizer
    train_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)
    #validate_loss_fn = nn.CrossEntropyLoss().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.schedule, gamma=0.1)
    # get train and val data loaders
    train_loader = get_data_loader(args=args, train=True)
    val_loader = get_data_loader(args=args, train=False)
    acc_val_max, acc_val_argmax, start_epoch = load_checkpoint(args, model, optimizer, scheduler)

    scheduler.step(args.start_epoch)
    print(f"Resuming from epoch {args.start_epoch}")
    print(f"Scheduler milestones: {scheduler.milestones}")
    print(f"Scheduler last_epoch: {scheduler.last_epoch}")
    print(f"Adjusted learning rate: {optimizer.param_groups[0]['lr']}")
    # for each epoch...
    for n_epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = optimizer.param_groups[0]['lr']
        print('Starting epoch {}/{},  learning_rate={}'.format(n_epoch + 1, args.epochs, current_learning_rate))
        
        # train
        #(loss_train, acc_train) = run_epoch(train=True, data_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, n_epoch=n_epoch, args=args, device=device)
        (loss_train, acc_train) = run_epoch(train=True, data_loader=train_loader, model=model, criterion=train_loss_fn, optimizer=optimizer, n_epoch=n_epoch, args=args, device=device)
        
        # validate
        (loss_val, acc_val) = run_epoch(train=False, data_loader=val_loader, model=model, criterion=criterion, optimizer=None, n_epoch=n_epoch, args=args, device=device)
        if (acc_val_max is None) or (acc_val > acc_val_max):
            acc_val_max = acc_val
            acc_val_argmax = n_epoch
            torch.save({"model_state_dict": model.state_dict(),"acc_val_max": acc_val_max,"acc_val_argmax": acc_val_argmax}, pathout + '/' + 'checkpoint_epoch{:>04d}_{:.2f}.pth'.format(n_epoch + 1,100.0 * acc_val_max))
        # adjust learning rate
        scheduler.step(n_epoch+1)
    
        # save the model weights
        #torch.save({"model_state_dict": model.state_dict()}, 'checkpoint_epoch{:>04d}.pth'.format(n_epoch + 1))
        
        # print epoch summary
        line = 'Epoch {}/{} summary:  loss_train={:.5f},  acc_train={:.2f}%,  loss_val={:.2f},  acc_val={:.2f}% (best: {:.2f}% @ epoch {})'.format(n_epoch + 1, args.epochs, loss_train, 100.0 * acc_train, loss_val, 100.0 * acc_val, 100.0 * acc_val_max, acc_val_argmax + 1)
        print('=' * len(line))
        print(line)
        print('=' * len(line))
        wA.writeLogAcc(filenameLOG,line)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped')
        sys.exit(0)
    except Exception as e:
        print('Error: {}'.format(e))
        sys.exit(1)
