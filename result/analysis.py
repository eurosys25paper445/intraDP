
# %%
from itertools import count
import os
import datetime
import numpy as np
from matplotlib import pyplot as plt
from csv import reader
import sys


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.min = float("inf")

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val < self.min:
            self.min = val

def preprocessing(line):
    lineData=line.strip().split(' ')
    return lineData

def read_file(path):
    print(f"reading file: {path}")
    f=open(path,"r")
    line = f.readline()  
    layers = [] 
    while line:              
        lineData=preprocessing(line) 
        if lineData[0]=="op:":
            each_op = {}
            each_op["type"]=lineData[1][:-1]
            each_op["index"]=int(lineData[3][:-1])
            if each_op["index"] == len(layers):
                each_op["input_size"] = AverageMeter()
                each_op["output_size"] = AverageMeter()
                each_op["computation_time"] = AverageMeter()
                layers.append(each_op)
            for i in range(len(lineData)):
                if lineData[i]=="input_size:":
                    layers[each_op["index"]]["input_size"].update(float(lineData[i+1]))                                               
            for i in range(len(lineData)):
                if lineData[i]=="output_size:":
                    layers[each_op["index"]]["output_size"].update(float(lineData[i+1]))  
        if  lineData[0]=="stamp": 
            if each_op["index"]==0 and layers[each_op["index"]]["computation_time"].count == 1:
                line = f.readline()
                continue
            layers[each_op["index"]]["computation_time"].update(float(lineData[6]))  
        line = f.readline() 
    # print(layers)
    return layers

def get_by_name(layers,case):
    result = []
    for op in layers:
        result.append(op[case])
    return result

def merge(robot,server):
    info=[]
    for i in range(len(robot)):
        each = {}
        each["type"]=robot[i]["type"]
        each["index"]=robot[i]["index"]
        each["input_size"]=robot[i]["input_size"].val
        each["output_size"]=robot[i]["output_size"].val
        each["robot"]=robot[i]["computation_time"].avg
        each["server"]=server[i]["computation_time"].avg
        if each["output_size"]< 1e-6:
            each["output_size"] = float('inf')
        if each["robot"]< 1e-6:
            each["robot"] = 1e-6
        if each["server"]< 1e-6:
            each["server"] = 1e-6
        info.append(each)
    return info

def total_computation_time(layer,where,start,end):
    total = 0.
    for i in range(start,end):
        total = total + layer[i][where]
    return total

def ParallelInference(info,bw,min_idx):
    best_info_size = info[0]["input_size"]
    transmit = 0.
    total_time=0.
    transmit_data = 0.
    for i in range(1,len(info)+1):
        transmit = transmit + (bw*info[i-1]['robot']/1000)/best_info_size
        transmit_data = transmit_data + bw*info[i-1]['robot']/1000
        total_time = total_time + info[i-1]['robot'] 
        # print(f"transmit {transmit}, time {total_time}")
        if transmit > 1.0:
            print(f"total transmit data is {transmit_data}, {transmit_data/info[0]['input_size']}")
            return total_time + total_computation_time(info,"server",i,len(info))
        if best_info_size > info[i-1]["output_size"]:
            best_info_size = info[i-1]["output_size"]
    return total_time

latency = -1.
def find_location(estimated):
    if latency < 0.:
        min_idx = 0
        min_time = estimated[0]["total"]
        for i in range(len(estimated)):
            if estimated[i]["total"]<min_time:
                min_time = estimated[i]["total"]
                min_idx = i
    else:
        min_idx = len(estimated)-1
        min_time = estimated[min_idx]["robot_time"]
        for i in range(len(estimated)):
            if estimated[i]["robot_time"]<min_time and estimated[i]["total"]<latency:
                min_time = estimated[i]["robot_time"]
                min_idx = i
    return min_idx



def estimate_inference_time(info,bw,model,diff_baseline,diff_PCI):
    estimated = []
    each = {}
    each["robot_time"] = 0.
    each["transmit_time"] = info[0]["input_size"]/bw*1000
    each["server_time"] = total_computation_time(info,"server",0,len(info))
    each["total"]=each["robot_time"] + each["transmit_time"] + each["server_time"]
    estimated.append(each)
    for i in range(1,len(info)+1):
        each = {}
        each["robot_time"] = total_computation_time(info,"robot",0,i)
        each["transmit_time"] = info[i-1]["output_size"]/bw*1000
        if i ==len(info):
            each["transmit_time"]=0.
        each["server_time"] = total_computation_time(info,"server",i,len(info))
        each["total"]=each["robot_time"] + each["transmit_time"] + each["server_time"]
        estimated.append(each)
    chosen_idx = find_location(estimated)
    # print(estimated)
    print(f"at bw {bw:.1f} MB/S, best location is {chosen_idx} {estimated[chosen_idx]['total']}")
    print(f"{estimated[chosen_idx]}")
    # print(f"{estimated[min_idx]['robot_time']/estimated[min_idx]['total']},{estimated[min_idx]['transmit_time']/estimated[min_idx]['total']},{estimated[min_idx]['server_time']/estimated[min_idx]['total']}")
    PCI_time = ParallelInference(info,bw,chosen_idx)
    diff_PCI.update(PCI_time/estimated[chosen_idx]['total'])
    print(f"at bw {bw:.1f} MB/S, best PCI is {PCI_time} {diff_PCI.val}")
    # for local in locations[model]:
    #     diff_baseline.update(estimated[local[0]]['total']/estimated[min_idx]['total']*100,local[1])
    #     diff_PCI.update(PCI_time/estimated[local[0]]['total']*100,local[1])
    #     # print(f"location {local[0]} {estimated[local[0]]}, vs baseline {diff_baseline.val}, vs PCI {diff_PCI.val}")

def analysis(model):
    robot_layers = read_file(os.path.join("./"+model+"_robot.txt"))
    server_layers = read_file(os.path.join("./"+model+"_server.txt"))
    info = merge(robot_layers,server_layers)
    diff_baseline = AverageMeter()
    diff_PCI = AverageMeter()
    for i in range(500):
        bw = 0.1+0.1*i
        estimate_inference_time(info,bw,model,diff_baseline,diff_PCI)
    print(f"average diff vs best PCI is {diff_PCI.avg}, max {diff_PCI.min}")

models=['vgg19',"vgg1916","kapao"]
models=["vgg1916"]
for model in models:
    analysis(model)