import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchsummaryX import summary
from mobilenet import MobileNetV1
from tqdm import tqdm
import os
import argparse

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validation(model, val_loader, loss_fn):
    acc1 = 0.0
    acc5 = 0.0
    losses = []
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(tqdm(val_loader)):
            input = data[0].cuda()
            gt = data[1].cuda()
            print(f'GT : {gt}')
            outputs = model(input)
            loss = loss_fn(outputs, gt)

            losses.append(loss)
            acc1_, acc5_ = accuracy(outputs, gt, topk=(1, 5))
            acc1 += acc1_
            acc5 += acc5_

    model.train()
    print(f'len(val_loader) : {len(val_loader)}')
    acc1 = acc1 / len(val_loader)
    acc5 = acc5 / len(val_loader)
    loss = sum(losses) / len(losses)

    return acc1, acc5, loss


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    config.distributed = config.world_size > 1 or config.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if config.multiprocessing_distributed:
        config.world_size = ngpus_per_node * config.world_size
        torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        main_worker(config.gpu, ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):
    if config.gpu is not None:
        print(f'Use GPU: {gpu} for training')

    if config.distributed:
        if config.dist_url == "envs://" and config.rank == -1:
            config.rank = int(os.environ["RANK"])
        if config.multiprocessing_distributed:
            config.rank = config.rank * ngpus_per_node + gpu
        torch.distributed.init_process_group(backend=config.dist_backend, init_method=config.dist_url, world_size=config.world_size, rank=config.rank)

    epochs = 100

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
                    transforms.RandomResizedCrop((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                    ])
    
    transform_val = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    normalize,
                    ])

    ImageNet_train = datasets.ImageFolder(config.data_path + '/train',transform = transform_train)
    ImageNet_valid = datasets.ImageFolder(config.data_path + '/val',transform = transform_val)

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(ImageNet_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(ImageNet_valid, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    config.batch_size = int(config.batch_size / ngpus_per_node)
    config.num_workers = int((config.num_workers + ngpus_per_node - 1) / ngpus_per_node)


    train_dataloader = DataLoader(ImageNet_train,batch_size=config.batch_size,num_workers=config.num_workers,pin_memory=True, sampler=train_sampler)
    valid_dataloader = DataLoader(ImageNet_valid,batch_size=1,num_workers=config.num_workers,pin_memory=True, sampler=val_sampler)

    model = MobileNetV1(ch_in=3, n_classes=1000).cuda(gpu)
    loss_fn = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)

        if epoch % 10 == 0:
            if torch.distributed.get_rank() == 0:
                print(f'Validation start')
                acc1, acc5, loss = validation(model, valid_dataloader, loss_fn)
                print(f'Epoch: {epoch}, acc1: {acc1}, acc5: {acc5}, loss: {loss}')
            torch.distributed.barrier()

        for (input, label) in tqdm(train_dataloader):
            input = input.cuda(gpu)
            label = label.cuda(gpu)
            output = model(input)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        lr_scheduler.step()

        if torch.distributed.get_rank() == 0:
            print(f'Saving checkpoint epoch {epoch}')
            torch.save(model.state_dict(), f'./checkpoint/epoch_{epoch}.pth')
        torch.distributed.barrier()


    print(f'Training done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_path', type=str, default='/home/lab-com/datasets/ImageNet1K/imagenet')
    ############ Distributed Data Parallel (DDP) ############
    parser.add_argument('--world_size', type=int, default=-1)
    parser.add_argument('--rank', type=int, default=-1)
    parser.add_argument('--gpu', type=str, default="0,1,2,3")
    parser.add_argument('--dist-url', type=str, default=f"tcp://localhost:23456")
    parser.add_argument('--dist-backend', type=str, default="nccl")
    parser.add_argument('--multiprocessing_distributed', default=True)
    
    config = parser.parse_args()
    print(config)

    if not os.path.exists('./checkpoint'):
        os.mkdir('./checkpoint')

    main(config)
