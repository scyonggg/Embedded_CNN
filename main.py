import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchsummaryX import summary
from mobilenet import MobileNetV1
from tqdm import tqdm


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

            outputs = model(input)
            loss = loss_fn(outputs, gt)

            losses.append(loss)
            acc1_, acc5_ = accuracy(outputs, gt, topk=(1, 5))
            acc1 += acc1_
            acc5 += acc5_

    model.train()

    acc1 = acc1 / len(val_loader)
    acc5 = acc5 / len(val_loader)
    loss = sum(losses) / len(losses)

    return acc1, acc5, loss


def main(model:torch.nn.Module):
    imagenet_path = '/home/lab-com/datasets/ImageNet1K/imagenet'
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


    ImageNet_train = datasets.ImageFolder(imagenet_path + '/train',transform = transform_train)
    ImageNet_valid = datasets.ImageFolder(imagenet_path + '/val',transform = transform_val)

    train_dataloader = DataLoader(ImageNet_train,batch_size=32,num_workers=4,pin_memory=True)
    valid_dataloader = DataLoader(ImageNet_valid,batch_size=1,num_workers=4,pin_memory=True)

    model = model.cuda()
    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(epochs):
        for (input, label) in tqdm(train_dataloader):
            input = input.cuda()
            label = label.cuda()
            output = model(input)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        lr_scheduler.step()

        if epoch % 10 == 0:
            print(f'Validation start')
            acc1, acc5, loss = validation(model, valid_dataloader, loss_fn)
            print(f'Epoch: {epoch}, acc1: {acc1}, acc5: {acc5}, loss: {loss}')

    print(f'Training done')

if __name__ == '__main__':
    model = MobileNetV1(ch_in=3, n_classes=1000)
    main(model)