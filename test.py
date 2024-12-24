import os
import json
import torch.optim as optim
import torch
from options import get_options
from utils import torch_load_cpu, load_problem, load_model, move_to
from nets.attention_model import AttentionModel
import pprint as pp
import torch.nn as nn
from train import get_inner_model
from torch.utils.data import DataLoader
import numpy as np
from datetime import timedelta
from utils.problem_augment import augment
import numpy as np
import itertools
from tqdm import tqdm
import time
import argparse
from utils.functions import parse_softmax_temperature



def eval_dataset(model, dataset_path, width, softmax_temp, opts, offset):
    # Even with multiprocessing, we load the model here since it contains the name where to write results

    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=opts.sample_size, offset=offset)
    results, best_cost, best_route, best_duration, max_val, start = _eval_dataset(model, dataset, width, softmax_temp, opts, device)
    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size
    # parallelism = num_processes
    costs, tours, durations = zip(*results)  # Not really costs since they should be negative
 
    return costs, durations, best_cost, best_route, best_duration,


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):

    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('greedy') else "sampling",
        temp=softmax_temp)
    
    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    best_cost = float('inf')
    best_route = None
    best_duration = None
    if opts.N_aug > 1:
        aug = opts.N_aug
    else:
        aug = 1

    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        if opts.problem == 'mtsp':
            max_val = batch.max()
            if max_val > 1:
                batch = batch/max_val
        else:
            max_val = None

        # For TSPLIB
        if aug > 1:
            batch = augment(batch, aug)

        # distance_matrix = torch.cdist(batch, batch, p=2)
        batch = move_to(batch, device)

        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy' and opts.N_aug == 8:
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                else:
                    batch_rep = width
                    iter_rep = 1
                print(model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep, agent_num=opts.agent_max, aug=aug))
                sequences, costs = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep, agent_num=opts.agent_max, aug=aug)
                

        duration = time.time() - start
        results.append((costs.cpu().numpy(), sequences.cpu().numpy(), duration))

        # Check if this batch contains a better solution
        batch_best_cost = costs.min().item()  # Get the best cost in this batch
        batch_best_index = costs.argmin().item()  # Get the index of the best route

        if batch_best_cost < best_cost:
            best_cost = batch_best_cost
            best_route = sequences[batch_best_index].cpu().numpy()  # Get the best route
            best_duration = duration

    return results, best_cost, best_route, best_duration, max_val, start

def run(opts):
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default="mtsp", type=str, help="problem type")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=100,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--decode_type', type=str, default='greedy',
                        help='Decode type, greedy or sampling')
    parser.add_argument('--width', type=int, nargs='+', default=[0],
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', type=str, default='greedy',
                        help='Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--multiprocessing', default=False,
                        help='Use multiprocessing to parallelize over multiple GPUs')
    parser.add_argument('--agent_num', default=8, type=int, help="decide the number of agent")
    parser.add_argument('--ft',default="Y", type=str)
    parser.add_argument('--is_serial', default='True', type=str, help="whether to use serial augmentation of instance")
    parser.add_argument('--N_aug', default=8, type=int, help="how any augmentation of instance")
    parser.add_argument('--max_calc_batch_size', default=100000, type=int, help="max batch size for calculation")
    opts = parser.parse_args()
    
    opts.problem = "mtsp"
    opts.graph_size = 505
    opts.agent_max = 8
    opts.device = torch.device("cuda:0" if opts.no_cuda else "cpu")
    problem = load_problem(opts.problem)
    opts.load_path = "pretrained/mtsp/mtsp500/epoch-0.pt"
    opts.model = "attention"
    opts.ft = "Y"
    opts.softmax_temperature = 1

    model = load_model(opts.load_path, agent_num=opts.agent_max, ft=opts.ft, epoch=1)

    # Prepare dataset for inference
    dataset_path = "test.tsp"

    opts.sample_size = 1
    opts.width = 0
    width = opts.width

    agent_num = opts.agent_max
    model, _ = load_model(opts.load_path, agent_num=agent_num, ft=opts.ft, epoch=1)
    model.agent_num = opts.agent_max

    print(eval_dataset(model, dataset_path, width, opts.softmax_temperature, opts, offset=None))

if __name__ == "__main__":
    opts = get_options()
    run(opts)
