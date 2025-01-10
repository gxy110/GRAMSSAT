"""
Replay SL to train surrogate top model
"""
from __future__ import print_function
import argparse
import os
import time
import ast
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import datasets.get_dataset as get_dataset
import matplotlib.pyplot as plt
import dill
import os
import subprocess
import math
from models import model_sets
import models.bottom_model_plus as models

import random
random.seed(37)
np.random.seed(37)
torch.manual_seed(37)

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def train_surrogate_model(smashed_data_path, pseudo_label_path, received_grad_path, proxy_model, optimizer, epoch_num, batch_num, batch_count, ssl_method):
    smashed_data = torch.from_numpy(np.load(smashed_data_path)).float().requires_grad_(True)
    received_grad = torch.from_numpy(np.load(received_grad_path)).float()
    pseudo_label = torch.from_numpy(np.load(pseudo_label_path)).long()  
    prediction = proxy_model(smashed_data)
    loss_prediction = F.cross_entropy(prediction, pseudo_label)
    loss_prediction.backward(create_graph=True) 
    loss_gradient_match = F.mse_loss(smashed_data.grad, received_grad, reduction='sum')

    # you can add other ssl methods
    if ssl_method == 'MixMatch':  
        # compute guessed labels of unlabel samples
        p = torch.softmax(prediction, dim=1)  
        pt = p ** (1 / args.T)  
        targets_u = pt / pt.sum(dim=1, keepdim=True)  
        targets_u = targets_u.detach() 

        # read labeled smashed data
        smashed_data_dir = args.load + "labeled_smashed_data/"
        labeled_smashed_data_path = os.path.join(smashed_data_dir, f"labeled_smashed_data_epoch_{epoch_num}_batch_{batch_num}.npy")
        inputs_x = torch.from_numpy(np.load(labeled_smashed_data_path)).float()
        # read corresponding labels
        targets_dir = args.load + "targets/"
        targets_path = os.path.join(targets_dir, f"targets_epoch_{epoch_num}_batch_{batch_num}.npy")
        targets_x = torch.from_numpy(np.load(targets_path)).long()

        # mixup  
        all_inputs = torch.cat([inputs_x, smashed_data], dim=0)
        all_targets = torch.cat([targets_x, targets_u], dim=0)

        l = np.random.beta(0.75, 0.75)
        l = max(l, 1 - l)
        idx = torch.randperm(all_inputs.size(0)) 
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b  
        mixed_target = l * target_a + (1 - l) * target_b

        batch_size = inputs_x.size(0)
        mixed_input = list(torch.split(mixed_input, batch_size)) 
        mixed_input = interleave(mixed_input, batch_size) 

        logits = [proxy_model(mixed_input[0])] 
        for input in mixed_input[1:]:
            logits.append(proxy_model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)  
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        # culculate MixMatch loss
        probs_u = torch.softmax(logits_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_target[:batch_size], dim=1))
        Lu = torch.mean((probs_u - mixed_target[batch_size:]) ** 2)
        w = args.lambda_u * linear_rampup(epoch_num + batch_num / batch_count, args.epochs)  # 对于lambda_u的取值我非常存疑
        ssl_loss = Lx + w * Lu

    else:
        ssl_loss = 0

    optimizer.zero_grad()
    total_loss = args.alpha*loss_prediction + args.beta*loss_gradient_match + ssl_loss
    total_loss.backward()  
    optimizer.step()

    return loss_prediction.item(), loss_gradient_match.item(), ssl_loss

def check_files_existence(paths):
    return all(os.path.exists(path) for path in paths)

def plot_and_annotate(ax, data, label, color):
    epochs = range(1, len(data) + 1)
    ax.plot(epochs, data, label=label, marker='o', color=color, linestyle='-')
    min_loss_value = min(data)
    min_loss_epoch = data.index(min_loss_value) + 1
    ax.scatter(min_loss_epoch, min_loss_value, color=color)
    ax.text(min_loss_epoch, min_loss_value, f'({min_loss_epoch}, {min_loss_value:.2f})', color=color)

def main():
    # folder_to_delete = f"./saved_experiment_results/saved_data/{args.dataset_name}_saved_data"
    # if os.path.exists(folder_to_delete):
    #     subprocess.run(['rm', '-rf', folder_to_delete], check=True)
    
    # write experiment setting into file name
    setting_str = ""
    setting_str += "_"
    setting_str += "lr="
    setting_str += str(args.lr)
    setting_str += "_"
    setting_str += "alpha="
    setting_str += str(args.alpha)
    setting_str += "_"
    setting_str += "beta="
    setting_str += str(args.beta)
    setting_str += "_"
    setting_str += "num_layer="
    setting_str += str(args.num_layer)
    setting_str += "_"
    setting_str += "activation_func="
    setting_str += str(args.activation_func_type)
    setting_str += "_"
    setting_str += "use_bn="
    setting_str += str(args.use_bn)
    setting_str += "_"
    setting_str += "idx="
    setting_str += str(args.idx)
    print(setting_str)

    # datasets settings
    dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset_name)
    train_samples_num = dataset_setup.train_samples_num
    batch_count = math.ceil(train_samples_num / args.batch_size)-1

    # initialize surrogate model
    print("==> creating surrogate top model")
    
    top_model = model_sets.TopModel(dataset_name=args.dataset_name).get_model(surrogate=True)
    
    cudnn.benchmark = True
    # print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # optimizer = optim.Adam(top_model.parameters(), lr=args.lr)
    optimizer = optim.SGD(top_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
   
    # Train and test
    stone1 = args.stone1  # 50 int(args.epochs * 0.5)
    stone2 = args.stone2  # 85 int(args.epochs * 0.8)
    lr_scheduler_top_model = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[stone1, stone2], gamma=args.step_gamma)
    
    dir_save_medium_data = args.load
    smashed_data_dir = dir_save_medium_data + "/smashed_data/"
    pseudo_label_dir = dir_save_medium_data + "/pseudo_labels/"
    gradient_dir = dir_save_medium_data + "/received_grads/"
    avg_pred_loss_per_epoch = []
    avg_grad_loss_per_epoch = []
    avg_ssl_loss_per_epoch = []
    avg_combine_loss_per_epoch = []
    for epoch in range(args.epochs):   
        read_epoch = epoch + (args.total_epochs-args.epochs)   
        avg_pred_loss = 0
        avg_grad_loss = 0
        avg_ssl_loss = 0
        while True:
            initial_files_exist = check_files_existence([
                os.path.join(smashed_data_dir, f"smashed_data_epoch_{read_epoch}_batch_0.npy"),
                os.path.join(pseudo_label_dir, f"pseudo_labels_epoch_{read_epoch}_batch_0.npy"),
                os.path.join(gradient_dir, f"received_grads_epoch_{read_epoch}_batch_0.npy")
            ])
            if initial_files_exist:
                break  
            else:
                print(f'Epoch {read_epoch}: lack of original files, wait for 5 seconds...')
                time.sleep(5)  

        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs-1, state['lr']))
        batch_num = 0
        while batch_num < batch_count:
            smashed_data_path = os.path.join(smashed_data_dir, f"smashed_data_epoch_{read_epoch}_batch_{batch_num}.npy")
            pseudo_label_path = os.path.join(pseudo_label_dir, f"pseudo_labels_epoch_{read_epoch}_batch_{batch_num}.npy")
            received_grad_path = os.path.join(gradient_dir, f"received_grads_epoch_{read_epoch}_batch_{batch_num}.npy")
            time.sleep(0.001)  

            if os.path.exists(smashed_data_path) and os.path.exists(pseudo_label_path) and os.path.exists(received_grad_path):
                loss_pred, loss_grad, loss_ssl = train_surrogate_model(smashed_data_path, pseudo_label_path, received_grad_path, top_model, optimizer, read_epoch, batch_num, batch_count, args.ssl_method)
                avg_pred_loss += loss_pred
                avg_grad_loss += loss_grad
                if args.ssl_method != 'None':
                    avg_ssl_loss += loss_ssl.detach().cpu().numpy()
                else:
                    avg_ssl_loss += 0

                # delete readed data if necessary
                # os.remove(smashed_data_path)
                # os.remove(pseudo_label_path)
                # os.remove(received_grad_path)

                batch_num += 1

                #print(f"Batch {batch_num}: Loss Prediction = {loss_pred}, Loss Gradient Match = {loss_grad}")

            else:
                print(f"Batch_idx: {batch_num}, Path: {smashed_data_path} is not found")
                time.sleep(0.5)

            if batch_num % 25 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_pred: {:.6f}, Loss_grad: {:.6f}'.format(
                        epoch, batch_num * args.batch_size, train_samples_num,
                            100. * batch_num / batch_count, loss_pred, loss_grad))
        
        lr_scheduler_top_model.step()

        avg_pred_loss /= batch_count
        avg_grad_loss /= batch_count
        avg_ssl_loss /= batch_count
        avg_combine_loss = args.alpha*avg_pred_loss + args.beta*avg_grad_loss + avg_ssl_loss
        avg_pred_loss_per_epoch.append(avg_pred_loss)
        avg_grad_loss_per_epoch.append(avg_grad_loss)
        avg_ssl_loss_per_epoch.append(avg_ssl_loss)
        avg_combine_loss_per_epoch.append(avg_combine_loss)

       
    # save model
    dir_save_surrogate_model = "./saved_experiment_results" + f"/saved_surrogate/{args.dataset_name}_saved_models"
    if not os.path.exists(dir_save_surrogate_model):
        os.makedirs(dir_save_surrogate_model)
    torch.save(top_model, os.path.join(dir_save_surrogate_model, f"{args.dataset_name}_saved_framework{setting_str}.pth"), pickle_module=dill)

    # draw the average loss per epoch
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    plot_and_annotate(ax, avg_pred_loss_per_epoch, 'Prediction Loss', 'blue')
    plot_and_annotate(ax, avg_grad_loss_per_epoch, 'Gradient Loss', 'green')
    plot_and_annotate(ax, avg_ssl_loss_per_epoch, str(args.ssl_method)+' Loss', 'yellow')
    plot_and_annotate(ax, avg_combine_loss_per_epoch, 'Combined Loss', 'red')
    combined_losses = avg_pred_loss_per_epoch + avg_grad_loss_per_epoch + avg_ssl_loss_per_epoch + avg_combine_loss_per_epoch
    y_max = max(combined_losses)
    y_min = min(combined_losses)
    padding = 0.1 * (y_max - y_min)
    ax.set_ylim([y_min - padding, y_max + padding])
    ax.legend()
    plt.title(f"Average Loss per Epoch of {args.dataset_name} in Surrogate Model")
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    pic_save_path = dir_save_surrogate_model + f"/{args.dataset_name}_saved_framework{setting_str}_loss_plot.png"
    plt.savefig(pic_save_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-train surrogate model')
    # dataset paras
    parser.add_argument('--dataset-name', default="Criteo", type=str,
                        choices=['CIFAR10', 'CIFAR100', 'TinyImageNet'])
    parser.add_argument('--dataset-path', default='./datasets/Datasets/CIFAR100', type=str)

    # surrogate paras
    parser.add_argument('--ssl-method', type=str, default=None, choices=['None','MixMatch'],
                    help='Choose an ssl method')
    parser.add_argument('--T', default=0.8, type=float, help='Temperature parameter of MixMatch')
    parser.add_argument('--lambda-u', default=50, type=float, help='Loss parameter of MixMatch')
    parser.add_argument('--num-layer', type=int, default=4,
                    help='number of layers of the inference head')
    parser.add_argument('--use-bn', type=ast.literal_eval, default=True,
                        help='Inference head use batchnorm or not')
    parser.add_argument('--activation_func_type', type=str, default='None',
                        help='Activation function type of the inference head',
                        choices=['ReLU', 'Sigmoid', 'None'])

    # sl paras
    parser.add_argument('--party-num', help='party-num',
                        type=int, default=2)
    parser.add_argument('--half', help='half number of features',
                        type=int, default=16)  # CIFAR10-16, TinyImageNet-32
    # checkpoints paras (used for trained bottom model in our attack)
    parser.add_argument('--load-dir',
                        default='./saved_experiment_results/saved_data/',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint', )

    # training paras
    parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                        help='train batchsize') 
    parser.add_argument('--total-epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run in original SL task') 
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run in surrogate train')  
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')  
    parser.add_argument('--alpha', default=0, type=float,
                        metavar='a', help='weight of prediction loss in total loss, which is not contained in mixed loss in our method')  
    parser.add_argument('--beta', default=1, type=float,
                        metavar='b', help='weight of gradient match loss in total loss')  
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--step-gamma', default=0.1, type=float, metavar='S',
                        help='gamma for step scheduler')
    parser.add_argument('--stone1', default=50, type=int, metavar='s1',
                        help='stone1 for step scheduler')
    parser.add_argument('--stone2', default=85, type=int, metavar='s2',
                        help='stone2 for step scheduler')
    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--idx', default=0, type=int,
                        help='the index of experiment')
    
    args = parser.parse_args()
    args.load = args.load_dir + f'test_{args.idx}/{args.dataset_name}_saved_data/'  
    state = {k: v for k, v in args._get_kwargs()}

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = torch.cuda.is_available()

    main()