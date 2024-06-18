import argparse
import os.path as osp
import shutil
import time
import warnings
import sys
import random

import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms, datasets, models


from ParallelCollaborativeInference import ParallelCollaborativeInference

tasks = ["all", "classification", "detection", "segmentation", "video"]
parser = argparse.ArgumentParser(description='torch vision inference')
parser.add_argument('-t', '--task', default='classification',
                    choices=tasks,
                    help='task: ' +
                    ' | '.join(tasks) +
                    ' (default: classification)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet101',
                    help='model architecture')
parser.add_argument('-d', '--dataset', default='CIFAR10',
                    help='dataset')
parser.add_argument('-p', '--parallel', default='select',
                    help='parallel approach')
parser.add_argument('-ip', '--ip', default='127.0.0.1',
                    help='server ip')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    args = parser.parse_args()
    task = args.task
    model_arch = args.arch
    dataset_name = args.dataset
    parallel_approach: str = args.parallel
    data_dir = osp.join(osp.dirname(osp.abspath(__file__)), "data", dataset_name)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using data dir: {data_dir}; parallel_approach {parallel_approach}")
    print(f"device {device}; task {task}; model_arch {model_arch}; dataset {dataset_name}")

    if False: # TODO support iterate through all these models and datasets
        if task == "all":
            all_models = models.list_models()
        if task == "classification":
            all_models = models.list_models(module=models)
        else:
            all_models = models.list_models(module=getattr(models, task))

        datasets = {
            "classification": ["CIFAR10", "CIFAR100", "MNIST"],
            "detection": ["CocoDetection", "Kitti", "VOCDetection"],
            "segmentation": ["Cityscapes", "VOCSegmentation"],
            "video": ["HMDB51", "Cityscapes"]
        }
    weights = getattr(models.get_model_weights(model_arch), "DEFAULT")
    preprocess = weights.transforms()
    model = models.get_model(model_arch, weights=weights)
    model.eval()
    model = model.to(device)

    ip = args.ip
    port = 12345
    PCI = ParallelCollaborativeInference(parallel_approach=parallel_approach, ip=ip, port=port)
    PCI.start_client(model=model, init_forward_count=1)
    PCI.offload_order(30)

    if dataset_name == "ImageNet":
        kwargs = {"split": "val"}
    elif "CIFAR" in dataset_name:
        kwargs = {"download": True, "train": False}

    dataset: datasets.DatasetFolder = getattr(datasets, dataset_name)(
        data_dir, transform=preprocess, **kwargs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
    inference_time=AverageMeter()
    for i, (inp, target) in enumerate(dataloader):
        stime = time.time()
        pred = model(inp.to(device))[0]
        inference_time.update(time.time() - stime)
        if task == "classification" and i % 100 == 0:
            print(f"Result: pred: {weights.meta['categories'][torch.argmax(pred)]}; gt: {dataset.classes[target]}.")
            print(f"inference time: {inference_time.val:.3f} ms, average {inference_time.avg:.3f} ms")
        if i > 20:
            break

