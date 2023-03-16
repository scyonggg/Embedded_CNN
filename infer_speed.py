import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import timm
import time
from tqdm import tqdm
import numpy as np
from mobilenet import MobileNetV1

class RandomDataset(Dataset):
    def __init__(self,  length, imsize):
        self.len = length
        self.data = torch.randn( 3, imsize, imsize, length)

    def __getitem__(self, index):
        return self.data[:,:,:,index]

    def __len__(self):
        return self.len




def main(modellist):
    batch_size = 4
    warm_up = 10
    num_test = 100

    rand_loader = DataLoader(dataset=RandomDataset(batch_size*(warm_up + num_test), 224),
                         batch_size=batch_size, shuffle=False,num_workers=8)

    def inference(modelname, benchmark, half=False):
        with torch.no_grad():
            if modelname == 'vgg16':
                model = timm.create_model(modelname,)
            elif modelname == 'mobilenet':
                model = MobileNetV1()
            elif modelname == 'googlenet':
                model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=False, init_weights=True)
            model=model.to('cuda')
            model.eval()
            precision = "float"
            durations = []
            print(f'Benchmarking Inference {modelname} ')
            for step,img in enumerate(rand_loader):
                img=getattr(img,precision)()
                torch.cuda.synchronize()
                start = time.time()
                model(img.to('cuda'))
                torch.cuda.synchronize()
                end = time.time()
                if step >= warm_up:
                    durations.append((end - start)*1000)
            print(f'{modelname} model average inference time : {sum(durations)/len(durations)}ms')
            
            if half:
                durations_half = []
                print(f'Benchmarking Inference half precision type {modelname} ')
                model.half()
                precision = "half"
                for step,img in enumerate(rand_loader):
                    img=getattr(img,precision)()
                    torch.cuda.synchronize()
                    start = time.time()
                    model(img.to('cuda'))
                    torch.cuda.synchronize()
                    end = time.time()
                    if step >= warm_up:
                        durations_half.append((end - start)*1000)
                print(f'{modelname} half model average inference time : {sum(durations_half)/len(durations_half)}ms')
                
            if half:
                benchmark[modelname] = {"fp32": np.mean(durations), "fp16": np.mean(durations_half)}
            else:
                benchmark[modelname] = {"fp32": np.mean(durations)}
        return benchmark

    benchmark = {}

    for arch in modellist:
        try:
            benchmark = inference(arch, benchmark)
        except:
            print("pass {}".format(arch))


if __name__ == '__main__':
    modellist=['vgg16', 'googlenet', 'mobilenet']
    main(modellist)
