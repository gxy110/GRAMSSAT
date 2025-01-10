import argparse
import ast
import os
import time
import dill
from time import time
import sys
sys.path.insert(0, "./")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from datasets import get_dataset
from my_utils import utils
from models import model_sets
import possible_defenses

plt.switch_backend('agg')

D_ = 2 ** 13
BATCH_SIZE = 1000

current_epoch = 0

import random
random.seed(37)
np.random.seed(37)
torch.manual_seed(37)


def split_data(data):
    if args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:32]
    elif args.dataset == 'TinyImageNet':
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:64]
    else:
        raise Exception('Unknown dataset name!')
    return x_a, x_b


def check_and_save_file(file_path, data, batch_idx, epoch_idx):
    dir_save_data = os.path.dirname(file_path)
    if not os.path.exists(dir_save_data):
        os.makedirs(dir_save_data)
    np.save(file_path+f"_epoch_{epoch_idx}_batch_{batch_idx}.npy", data)
    return file_path


class SLTrain(nn.Module):
    def __init__(self):
        super(SLTrain, self).__init__()

        # counter for direct label inference attack
        self.inferred_correct = 0
        self.inferred_wrong = 0

        # attacker knowledges
        self.num_classes = None  # get the number of classes from dataset_setup in set_loaders()
        self.dir_save_pseudo_labels = args.save_dir + f"/saved_data/test_{args.idx}/{args.dataset}_saved_data/pseudo_labels/pseudo_labels"

        # collect data
        self.collect_top_inputs = True 
        self.top_inputs = torch.tensor([]).cuda()

        self.collect_grads = True
        self.grads = torch.tensor([]).cuda()

        # geneate pseudo labels
        self.cul_similarity = self.l2_distance

        # In order to evaluate attack performance, we need to collect label sequence of training dataset
        self.labels_training_dataset = torch.tensor([], dtype=torch.long).cuda()
        # whether to use real labels to test
        self.if_collect_training_dataset_labels = False

        # adversarial options
        self.defense_ppdl = args.ppdl
        self.defense_gc = args.gc
        self.defense_lap_noise = args.lap_noise
        self.defense_multistep_grad = args.multistep_grad

        # loss funcs
        self.loss_func_top_model = nn.CrossEntropyLoss()
        self.loss_func_bottom_model = utils.keep_predict_loss

        # we only consider two party split learning in GRAMSSAT,
        # concatenate the outputs of two bottom models to simulate a single client attacker.
        # bottom model A
        self.malicious_bottom_model_a = model_sets.BottomModel(dataset_name=args.dataset).get_model(
            half=args.half,
            is_adversary=True
        )
        # bottom model B
        self.benign_bottom_model_b = model_sets.BottomModel(dataset_name=args.dataset).get_model(
            half=args.half,
            is_adversary=False
        )
        # top model
        self.top_model = model_sets.TopModel(dataset_name=args.dataset).get_model()
        self.optimizer_top_model = optim.SGD(self.top_model.parameters(),
                                                 lr=args.lr,
                                                 momentum=args.momentum,
                                                 weight_decay=args.weight_decay)
        self.optimizer_malicious_bottom_model_a = optim.SGD(self.malicious_bottom_model_a.parameters(),
                                                                lr=args.lr, momentum=args.momentum,
                                                                weight_decay=args.weight_decay)
        self.optimizer_benign_bottom_model_b = optim.SGD(self.benign_bottom_model_b.parameters(),
                                                                lr=args.lr,
                                                                momentum=args.momentum,
                                                                weight_decay=args.weight_decay)
        
    def forward(self, x):
        # in vertical federated setting, each party has non-lapping features of the same sample
        x_a, x_b = split_data(x)
        out_a = self.malicious_bottom_model_a(x_a)
        out_b = self.benign_bottom_model_b(x_b)
        out = self.top_model(out_a, out_b)
        return out

    def l2_distance(self, a, b):
        a = a.view(a.shape[0], -1)  
        b = b.view(b.shape[0], -1)  

        diff = a[:, None] - b  
        dist = torch.norm(diff, p=2, dim=2)  

        return dist

    def generate_pseudo_hard_labels(self, similarity, labeled_labels_tensor, batch_idx, epoch_idx):
        indices = similarity.argmin(dim=1)  
        pseudo_labels = labeled_labels_tensor[indices] 
        pseudo_labels_cpu = pseudo_labels.detach().cpu().numpy()
        file_path = check_and_save_file(self.dir_save_pseudo_labels, pseudo_labels_cpu, batch_idx, epoch_idx)
        return file_path

    def simulate_train_round_per_batch(self, data, target, batch_idx, labeled_dataset, epoch_idx, inputs_x, targets_x):
        timer_mal = 0
        timer_benign = 0
        # simulate: bottom models forward, top model forward, top model backward and update, bottom backward and update
        
        # store grad of input of top model/outputs of bottom models
        input_tensor_top_model_a = torch.tensor([], requires_grad=True)
        input_tensor_top_model_b = torch.tensor([], requires_grad=True)

        # --bottom models forward--
        x_a, x_b = split_data(data)
        inputs_x_a, inputs_x_b = split_data(inputs_x)
        inputs_x_a = inputs_x_a.type(torch.float)
        inputs_x_b = inputs_x_b.type(torch.float)
        targets_x = targets_x.view(-1, 1).type(torch.long)
        targets_x = torch.zeros(args.batch_size, self.num_classes).scatter_(1, targets_x, 1)
        targets_x = targets_x.cuda(non_blocking=True)
        inputs_x_a, inputs_x_b= inputs_x_a.cuda(), inputs_x_b.cuda()

        labeled_samples = []
        labeled_labels = []

        for image, label in labeled_dataset:
            labeled_samples.append(image)
            labeled_labels.append(label)  

        labeled_samples_tensor = torch.stack(labeled_samples)  
        labeled_labels_tensor = torch.tensor(labeled_labels)  

        if torch.cuda.is_available():
            labeled_samples_tensor = labeled_samples_tensor.cuda()
            labeled_labels_tensor = labeled_labels_tensor.cuda()

        x_a_labeled, x_b_labeled = split_data(labeled_samples_tensor)

        # -bottom model A-
        self.malicious_bottom_model_a.train(mode=True)
        start = time()
        output_tensor_bottom_model_a = self.malicious_bottom_model_a(x_a)
        labeled_output_a = self.malicious_bottom_model_a(inputs_x_a)
        end = time()
        time_cost = end - start
        timer_mal += time_cost

        # -bottom model B-
        self.benign_bottom_model_b.train(mode=True)
        start2 = time()
        output_tensor_bottom_model_b = self.benign_bottom_model_b(x_b)
        labeled_output_b = self.benign_bottom_model_b(inputs_x_b)
        end2 = time()
        time_cost2 = end2 - start2
        timer_benign += time_cost2

        completed_inputs_x = torch.cat((labeled_output_a, labeled_output_b), dim=1).detach().cpu().numpy()
        dir_save_labeled_smashed_data = args.save_dir + f"/saved_data/test_{args.idx}/{args.dataset}_saved_data/labeled_smashed_data/labeled_smashed_data"
        file_path = check_and_save_file(dir_save_labeled_smashed_data, completed_inputs_x, batch_idx, epoch_idx)
        
        targets_x.view(-1, 1).type(torch.long)
        targets_x_cpu = targets_x.detach().cpu().numpy()
        dir_save_targets = args.save_dir + f"/saved_data/test_{args.idx}/{args.dataset}_saved_data/targets/targets"
        file_path = check_and_save_file(dir_save_targets, targets_x_cpu, batch_idx, epoch_idx)

        # -top model-
        # (we omit interactive layer for it doesn't effect our attack or possible defenses)
        # by concatenating output of bottom a/b(dim=10+10=20), we get input of top model
        input_tensor_top_model_a.data = output_tensor_bottom_model_a.data
        input_tensor_top_model_b.data = output_tensor_bottom_model_b.data

        if self.collect_top_inputs:
            output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
            self.top_inputs = torch.cat((self.top_inputs, output_bottom_models.data))
            top_inputs_cpu = self.top_inputs.cpu().numpy()
            dir_save_smashed_data = args.save_dir + f"/saved_data/test_{args.idx}/{args.dataset}_saved_data/smashed_data/smashed_data"
            file_path = check_and_save_file(dir_save_smashed_data, top_inputs_cpu, batch_idx, epoch_idx)
            #print(f"Smashed data saved at: {file_path}")  
            self.top_inputs = torch.empty(0, dtype=self.top_inputs.dtype, device=self.top_inputs.device)
        
        self.top_model.train(mode=True)
        output_framework = self.top_model(input_tensor_top_model_a, input_tensor_top_model_b)
        # --top model backward/update--
        loss_framework = model_sets.update_top_model_one_batch(optimizer=self.optimizer_top_model,
                                                                model=self.top_model,
                                                                output=output_framework,
                                                                batch_target=target,
                                                                loss_func=self.loss_func_top_model)

        # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
        grad_output_bottom_model_a = input_tensor_top_model_a.grad
        grad_output_bottom_model_b = input_tensor_top_model_b.grad

        # defenses here: the server(who controls top model) can defend against label inference attack by protecting
        # gradients sent to bottom models
        model_all_layers_grads_list = [grad_output_bottom_model_a, grad_output_bottom_model_b]
        # privacy preserving deep learning
        if self.defense_ppdl:
            possible_defenses.dp_gc_ppdl(epsilon=1.8, sensitivity=1, layer_grad_list=[grad_output_bottom_model_a],
                                         theta_u=args.ppdl_theta_u, gamma=0.001, tau=0.0001)
            possible_defenses.dp_gc_ppdl(epsilon=1.8, sensitivity=1, layer_grad_list=[grad_output_bottom_model_b],
                                         theta_u=args.ppdl_theta_u, gamma=0.001, tau=0.0001)
        # gradient compression
        if self.defense_gc:
            tensor_pruner = possible_defenses.TensorPruner(zip_percent=args.gc_preserved_percent)
            for tensor_id in range(len(model_all_layers_grads_list)):
                tensor_pruner.update_thresh_hold(model_all_layers_grads_list[tensor_id])
                # print("tensor_pruner.thresh_hold:", tensor_pruner.thresh_hold)
                model_all_layers_grads_list[tensor_id] = tensor_pruner.prune_tensor(
                    model_all_layers_grads_list[tensor_id])
        # differential privacy
        if self.defense_lap_noise:
            dp = possible_defenses.DPLaplacianNoiseApplyer(beta=args.noise_scale)
            for tensor_id in range(len(model_all_layers_grads_list)):
                model_all_layers_grads_list[tensor_id] = dp.laplace_mech(model_all_layers_grads_list[tensor_id])
        # multistep gradient
        if self.defense_multistep_grad:
            for tensor_id in range(len(model_all_layers_grads_list)):
                model_all_layers_grads_list[tensor_id] = possible_defenses.multistep_gradient(
                    model_all_layers_grads_list[tensor_id], bins_num=args.multistep_grad_bins,
                    bound_abs=args.multistep_grad_bound_abs)
       
        grad_output_bottom_model_a, grad_output_bottom_model_b = tuple(model_all_layers_grads_list)
       
        # collect gradients from top model
        if self.collect_grads:
            grad_output_bottom_models = torch.cat((grad_output_bottom_model_a, grad_output_bottom_model_b), dim=1)
            self.grads = torch.cat((self.grads, grad_output_bottom_models.data))
            grads_cpu = self.grads.cpu().numpy()
            dir_save_collect_grads = args.save_dir + f"/saved_data/test_{args.idx}/{args.dataset}_saved_data/received_grads/received_grads"
            file_path = check_and_save_file(dir_save_collect_grads, grads_cpu, batch_idx, epoch_idx)
            #print(f"Received grads saved at: {file_path}")
            self.grads = torch.empty(0, dtype=self.grads.dtype, device=self.grads.device)

        # generate pseudo labels for unlabeled data per batch
        # use real labels as pseudo labels for testing
        if self.if_collect_training_dataset_labels: 
            self.labels_training_dataset = torch.cat((self.labels_training_dataset, target), dim=0)
            pseudo_labels_cpu = self.labels_training_dataset.cpu().numpy()
            file_path = check_and_save_file(self.dir_save_pseudo_labels, pseudo_labels_cpu, batch_idx, epoch_idx)
            self.labels_training_dataset = torch.empty(0, dtype=self.labels_training_dataset.dtype, device=self.labels_training_dataset.device)
        # use smashed data to generate pseudo labels
        else: 
            labeled_output_tensor_bottom_model_a = self.malicious_bottom_model_a(x_a_labeled)
            labeled_output_tensor_bottom_model_b = self.benign_bottom_model_b(x_b_labeled)
            concatenated_features = torch.cat((output_tensor_bottom_model_a, output_tensor_bottom_model_b), dim=1)
            labeled_concatenated_features = torch.cat((labeled_output_tensor_bottom_model_a, labeled_output_tensor_bottom_model_b), dim=1)
            similarity = self.cul_similarity(concatenated_features, labeled_concatenated_features)
            file_path = self.generate_pseudo_hard_labels(similarity, labeled_labels_tensor, batch_idx, epoch_idx)
        
        # --bottom models backward/update--
        start = time()
        
        # --bottom models backward/update--
        # -bottom model a: backward/update-
        # print("malicious_bottom_model_a")
        start = time()
        model_sets.update_bottom_model_one_batch(optimizer=self.optimizer_malicious_bottom_model_a,
                                                model=self.malicious_bottom_model_a,
                                                output=output_tensor_bottom_model_a,
                                                batch_target=grad_output_bottom_model_a,
                                                loss_func=self.loss_func_bottom_model)
        end = time()
        time_cost = end - start
        timer_mal += time_cost
        # -bottom model b: backward/update-
        # print("benign_bottom_model_b")
        model_sets.update_bottom_model_one_batch(optimizer=self.optimizer_benign_bottom_model_b,
                                                model=self.benign_bottom_model_b,
                                                output=output_tensor_bottom_model_b,
                                                batch_target=grad_output_bottom_model_b,
                                                loss_func=self.loss_func_bottom_model)
        end2 = time()
        time_cost2 = end2 - end
        timer_benign += time_cost2
        timer_on = False
        if timer_on:
            print("timer_mal:", timer_mal)
            print("timer_benign:", timer_benign)
        return loss_framework


def correct_counter(output, target, topk=(1, 5)):
    correct_counts = []
    for k in topk:
        _, pred = output.topk(k, 1, True, True)
        correct_k = torch.eq(pred, target.view(-1, 1)).sum().float().item()
        correct_counts.append(correct_k)
    return correct_counts


def test_per_epoch(test_loader, framework, k=5, loss_func_top_model=None):
    test_loss = 0
    correct_top1 = 0
    correct_topk = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.dataset == 'Yahoo':
                for i in range(len(data)):
                    data[i] = data[i].long().cuda()
                target = target[0].long().cuda()
            else:
                data = data.float().cuda()
                target = target.long().cuda()
            # set all sub-models to eval mode.
            framework.malicious_bottom_model_a.eval()
            framework.benign_bottom_model_b.eval()
            framework.top_model.eval()
            # run forward process of the whole framework
            x_a, x_b = split_data(data)
            output_tensor_bottom_model_a = framework.malicious_bottom_model_a(x_a)
            output_tensor_bottom_model_b = framework.benign_bottom_model_b(x_b)

            output_framework = framework.top_model(output_tensor_bottom_model_a, output_tensor_bottom_model_b)
            
            correct_top1_batch, correct_topk_batch = correct_counter(output_framework, target, (1, k))

            # sum up batch loss
            test_loss += loss_func_top_model(output_framework, target).data.item()

            correct_top1 += correct_top1_batch
            correct_topk += correct_topk_batch
            # print("one batch done")
            count += 1
            if int(0.1 * len(test_loader)) > 0:
                count_percent_10 = count // int(0.1 * len(test_loader))
                if count_percent_10 <= 10 and count % int(0.1 * len(test_loader)) == 0 and\
                        count // int(0.1 * len(test_loader)) > 0:
                    print(f'{count // int(0.1 * len(test_loader))}0 % completed...')
                # print(count)

            if args.dataset == 'Criteo' and count == test_loader.train_batches_num:
                break

        if args.dataset == 'Criteo':
            num_samples = len(test_loader) * BATCH_SIZE
        else:
            num_samples = len(test_loader.dataset)
        test_loss /= num_samples
        print('Loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%), Top {} Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss,
            correct_top1, num_samples, 100.00 * float(correct_top1) / num_samples,
            k,
            correct_topk, num_samples, 100.00 * float(correct_topk) / num_samples
        ))


def set_loaders():
    dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset)
    num_classes = dataset_setup.num_classes
    train_dataset = dataset_setup.get_transformed_dataset(args.path_dataset, None, True)
    test_dataset = dataset_setup.get_transformed_dataset(args.path_dataset, None, False)
    labeled_dataset, _ = dataset_setup.get_transformed_labeled_dataset(1, args.path_dataset)  # only need 1 labeled data per class
    labeled_dataset_for_ssl, _ = dataset_setup.get_transformed_labeled_dataset(args.n_labeled_per_class_for_ssl, args.path_dataset)

    current_seed_callback = lambda: utils.get_seed_for_current_epoch(current_epoch)
    train_sampler = utils.CustomRandomSampler(train_dataset, seed_callback = current_seed_callback)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size, #shuffle=True, 
        sampler = train_sampler,  
        # num_workers=args.workers
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        # num_workers=args.workers
    )
    
    return train_loader, test_loader, num_classes, labeled_dataset, labeled_dataset_for_ssl


def main():
    global current_epoch   

    # write experiment setting into file name
    setting_str = ""
    setting_str += "_"
    setting_str += "lr="
    setting_str += str(args.lr)
    if args.ppdl:
        setting_str += "_"
        setting_str += "ppdl-theta_u="
        setting_str += str(args.ppdl_theta_u)
    if args.gc:
        setting_str += "_"
        setting_str += "gc-preserved_percent="
        setting_str += str(args.gc_preserved_percent)
    if args.lap_noise:
        setting_str += "_"
        setting_str += "lap_noise-scale="
        setting_str += str(args.noise_scale)
    if args.multistep_grad:
        setting_str += "_"
        setting_str += "multistep_grad_bins="
        setting_str += str(args.multistep_grad_bins)

    if args.n_labeled_per_class_for_ssl:
        setting_str += "_n="
        setting_str += str(args.n_labeled_per_class_for_ssl)
    if args.idx:
        setting_str += "_idx="
        setting_str += str(args.idx)
    print("settings:", setting_str)

    model = SLTrain()
    model = model.cuda()
    cudnn.benchmark = True

    stone1 = args.stone1  # 50 int(args.epochs * 0.5)
    stone2 = args.stone2  # 85 int(args.epochs * 0.8)
    lr_scheduler_top_model = torch.optim.lr_scheduler.MultiStepLR(model.optimizer_top_model,
                                                                  milestones=[stone1, stone2], gamma=args.step_gamma)
    lr_scheduler_m_a = torch.optim.lr_scheduler.MultiStepLR(model.optimizer_malicious_bottom_model_a,
                                                            milestones=[stone1, stone2], gamma=args.step_gamma)
    lr_scheduler_b_b = torch.optim.lr_scheduler.MultiStepLR(model.optimizer_benign_bottom_model_b,
                                                            milestones=[stone1, stone2], gamma=args.step_gamma)

    train_loader, val_loader, num_classes, labeled_dataset, labeled_dataset_for_ssl = set_loaders()
    labeled_trainloader = torch.utils.data.DataLoader(labeled_dataset_for_ssl, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    labeled_train_iter = iter(labeled_trainloader)
    
    model.num_classes = num_classes

    dir_save_model = args.save_dir + f"/saved_models/{args.dataset}_saved_models/surrogate_test"
    if not os.path.exists(dir_save_model):
        os.makedirs(dir_save_model)

    # start training. do evaluation every epoch.
    avg_loss_per_epoch = []
    for epoch in range(args.epochs):
        current_epoch = epoch 
        avg_loss = 0
        batch_num = 0
        print('model.optimizer_top_model current lr {:.5e}'.format(model.optimizer_top_model.param_groups[0]['lr']))
        print('model.optimizer_malicious_bottom_model_a current lr {:.5e}'.format(
            model.optimizer_malicious_bottom_model_a.param_groups[0]['lr']))
        print('model.optimizer_benign_bottom_model_b current lr {:.5e}'.format(
            model.optimizer_benign_bottom_model_b.param_groups[0]['lr']))

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.float().cuda()
            target = target.long().cuda()
            try:
                inputs_x, targets_x = labeled_train_iter.next()
            except StopIteration:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_train_iter.next()
            loss_framework = model.simulate_train_round_per_batch(data, target, batch_idx, labeled_dataset, epoch, inputs_x, targets_x)
            avg_loss += loss_framework
            batch_num += 1
            if batch_idx % 25 == 0:
                num_samples = len(train_loader.dataset)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), num_samples,
                           100. * batch_idx / len(train_loader), loss_framework.data.item()))
        avg_loss /= batch_num
        avg_loss_per_epoch.append(avg_loss.detach().cpu().numpy())
        lr_scheduler_top_model.step()
        lr_scheduler_m_a.step()
        lr_scheduler_b_b.step()

        if epoch == args.epochs - 1:
            txt_name = f"{args.dataset}_saved_framework{setting_str}"
            savedStdout = sys.stdout
            with open(dir_save_model + '/' + txt_name + '.txt', 'w+') as file:
                sys.stdout = file
                print('Evaluation on the training dataset:')
                test_per_epoch(test_loader=train_loader, framework=model, k=args.k,
                               loss_func_top_model=model.loss_func_top_model)
                print('Evaluation on the testing dataset:')
                test_per_epoch(test_loader=val_loader, framework=model, k=args.k,
                               loss_func_top_model=model.loss_func_top_model)
                sys.stdout = savedStdout
            print('Last epoch evaluation saved to txt!')

        print('Evaluation on the training dataset:')
        test_per_epoch(test_loader=train_loader, framework=model, k=args.k,
                       loss_func_top_model=model.loss_func_top_model)
        print('Evaluation on the testing dataset:')
        test_per_epoch(test_loader=val_loader, framework=model, k=args.k, loss_func_top_model=model.loss_func_top_model)

    # save model
    torch.save(model, os.path.join(dir_save_model, f"{args.dataset}_saved_framework{setting_str}.pth"),
               pickle_module=dill)
    
    # draw loss 
    pic_save_path = dir_save_model + f"/{args.dataset}_saved_framework{setting_str}_loss_plot.png"
    min_loss_value = min(avg_loss_per_epoch)
    min_loss_epoch = avg_loss_per_epoch.index(min_loss_value) + 1 
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs+1), avg_loss_per_epoch, marker='o', linestyle='-', color='blue')
    plt.scatter(min_loss_epoch, min_loss_value, color='red') 
    plt.text(min_loss_epoch, min_loss_value, f'({min_loss_epoch}, {min_loss_value:.2f})', color='red')  
    plt.ylim([min_loss_value - 0.1 * min_loss_value, max(avg_loss_per_epoch) + 0.1 * max(avg_loss_per_epoch)])
    plt.title(f"Average Loss per Epoch of {args.dataset}")
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig(pic_save_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vfl framework training')
    # dataset paras
    parser.add_argument('-d', '--dataset', default='CIFAR100', type=str,
                        help='name of dataset',
                        choices=['CIFAR10', 'CIFAR100', 'TinyImageNet'])
    parser.add_argument('--path-dataset', help='path_dataset',
                        type=str, default='./datasets/Datasets/CIFAR100')
    # knowledge paras
    parser.add_argument('--n-labeled-per-class-for-ssl', help='numbers of known labeled data of each class for ssl()',
                        type=int, default=4)
    # framework paras
    parser.add_argument('--half', help='half number of features',
                        type=int,
                        default=16)  # CIFAR10-16, TinyImageNet-32
    # evaluation & visualization paras
    parser.add_argument('--k', help='top k accuracy',
                        type=int, default=5)
    # saving path paras
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models and csv files',
                        default='./saved_experiment_results', type=str)
    # possible defenses on/off paras
    parser.add_argument('--ppdl', help='turn_on_privacy_preserving_deep_learning',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--gc', help='turn_on_gradient_compression',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--lap-noise', help='turn_on_lap_noise',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--multistep_grad', help='turn on multistep-grad',
                        type=ast.literal_eval, default=False)
    # paras about possible defenses
    parser.add_argument('--ppdl-theta-u', help='theta-u parameter for defense privacy-preserving deep learning',
                        type=float, default=1)
    parser.add_argument('--gc-preserved-percent', help='preserved-percent parameter for defense gradient compression',
                        type=float, default=0.1)
    parser.add_argument('--noise-scale', help='noise-scale parameter for defense noisy gradients',
                        type=float, default=0.005)
    parser.add_argument('--multistep_grad_bins', help='number of bins in multistep-grad',
                        type=int, default=6)
    parser.add_argument('--multistep_grad_bound_abs', help='bound of multistep-grad',
                        type=float, default=3e-2)
    # training paras
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of datasets loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')
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

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = torch.cuda.is_available()

    main()
