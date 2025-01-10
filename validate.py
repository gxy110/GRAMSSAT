from __future__ import print_function
import argparse
import ast
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import datasets.get_dataset as get_dataset
import dill
import copy
from models import model_sets
import models.bottom_model_plus as models
import os

D_ = 2 ** 13

class BottomPlusSurrogateModel(nn.Module):
    def __init__(self):
        super(BottomPlusSurrogateModel, self).__init__()
        self.loss_func_top_model = nn.CrossEntropyLoss()

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
        dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset)
        num_classes = dataset_setup.num_classes
        size_bottom_out = dataset_setup.size_bottom_out * 2
        self.surrogate_top_model = models.SurrogateTopModel(size_bottom_out, num_classes,
                                    num_layer=args.num_layer,
                                    activation_func_type=args.activation_func_type,
                                    use_bn=args.use_bn)
def split_data(data):
    if args.dataset in ['CIFAR10', 'CIFAR100']:
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:32]
    elif args.dataset == 'TinyImageNet':
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:64]
    else:
        raise Exception('Unknown dataset name!')
    return x_a, x_b

def set_loaders():
    dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset)
    train_dataset = dataset_setup.get_transformed_dataset(args.dataset_path, None, True)
    test_dataset = dataset_setup.get_transformed_dataset(args.dataset_path, None, False)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size, 
        shuffle = True, 
        # num_workers=args.workers
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        # num_workers=args.workers
    )
    return train_loader, test_loader

def correct_counter(output, target, topk=(1, 5)):
    correct_counts = []
    for k in topk:
        _, pred = output.topk(k, 1, True, True)
        correct_k = torch.eq(pred, target.view(-1, 1)).sum().float().item()
        correct_counts.append(correct_k)
    return correct_counts

def validate(test_loader, framework, k=5, loss_func_top_model=None):
    framework = framework.to('cuda')
    test_loss = 0
    correct_top1 = 0
    correct_topk = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.float().cuda()
            target = target.long().cuda()
            # set all sub-models to eval mode.
            framework.malicious_bottom_model_a.eval()
            framework.benign_bottom_model_b.eval()
            framework.surrogate_top_model.eval()
            # run forward process of the whole framework
            x_a, x_b = split_data(data)
            output_tensor_bottom_model_a = framework.malicious_bottom_model_a(x_a)
            output_tensor_bottom_model_b = framework.benign_bottom_model_b(x_b)
            if args.use_real_top:
                output_framework = framework.surrogate_top_model(output_tensor_bottom_model_a, output_tensor_bottom_model_b)
            else:
                output_bottom_models = torch.cat((output_tensor_bottom_model_a, output_tensor_bottom_model_b), dim=1).cuda()
                output_framework = framework.surrogate_top_model(output_bottom_models)

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

        num_samples = len(test_loader.dataset)
        test_loss /= num_samples
        print('Loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%), Top {} Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss,
            correct_top1, num_samples, 100.00 * float(correct_top1) / num_samples,
            k,
            correct_topk, num_samples, 100.00 * float(correct_topk) / num_samples
        ))


def main():
    model = BottomPlusSurrogateModel()
    model = model.cuda()
    cudnn.benchmark = True

    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    bottom_models_resume_path = args.resume_bottom
    checkpoint_bottom_models = torch.load(bottom_models_resume_path, pickle_module=dill)

    print("checkpoint_ba:", checkpoint_bottom_models.malicious_bottom_model_a)
    model.malicious_bottom_model_a = copy.deepcopy(checkpoint_bottom_models.malicious_bottom_model_a)
    model.benign_bottom_model_b = copy.deepcopy(checkpoint_bottom_models.benign_bottom_model_b)
    
    surrogate_model_resume_path = args.resume_surrogate
    if args.use_real_top:
        print(args.use_real_top)
        print("checkpoint_rt:", checkpoint_bottom_models.top_model)
        model.surrogate_top_model = copy.deepcopy(checkpoint_bottom_models.top_model)
    else:
        checkpoint_surrogate_model = torch.load(surrogate_model_resume_path, pickle_module=dill)
        model.surrogate_top_model = copy.deepcopy(checkpoint_surrogate_model)
        print("checkpoint_st:", checkpoint_surrogate_model)

    train_loader, val_loader = set_loaders()

    print('Evaluation on the training dataset:')
    validate(test_loader=train_loader, framework=model, k=args.k,
                    loss_func_top_model=model.loss_func_top_model)
    print('Evaluation on the testing dataset:')
    validate(test_loader=val_loader, framework=model, k=args.k,
                    loss_func_top_model=model.loss_func_top_model)

   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vfl framework training')
    # dataset paras
    parser.add_argument('--dataset', default="CIFAR100", type=str,
                        choices=['CIFAR10', 'CIFAR100', 'TinyImageNet'])
    parser.add_argument('--dataset-path', default='./datasets/Datasets/CIFAR100', type=str)

    # sl paras
    parser.add_argument('--num-epochs', default=100, type=int)
    parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                        help='train batchsize')  
    parser.add_argument('--party-num', help='party-num',
                        type=int, default=2)
    parser.add_argument('--half', help='half number of features',
                        type=int, default=16)  # CIFAR10-16, TinyImageNet-32
    parser.add_argument('--k', help='top k accuracy',
                        type=int, default=5)
    
    # surrogate paras
    parser.add_argument('--num-layer', type=int, default=4,
                    help='number of layers of the inference head')
    parser.add_argument('--use-bn', type=ast.literal_eval, default=True,
                        help='Inference head use batchnorm or not')
    parser.add_argument('--activation_func_type', type=str, default='None',
                        help='Activation function type of the inference head',
                        choices=['ReLU', 'Sigmoid', 'None'])
    
    # checkpoints paras (used for trained bottom model in our attack)
    parser.add_argument('--resume-dir',
                        default='./saved_experiment_results',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint', )
    parser.add_argument('--resume-name-bottom',
                        default='saved_experiment_results/saved_models/CIFAR100_saved_models/surrogate_test/CIFAR100_saved_framework_lr=0.001_n=4_idx=1.pth',
                        type=str, metavar='NAME',
                        help='file name of the latest checkpoint', )
    parser.add_argument('--resume-name-surrogate',
                        default='....',
                        type=str, metavar='NAME',
                        help='file name of the latest checkpoint', )
    parser.add_argument('--use-real-top',
                        default=False,
                        type=bool, 
                        help='whether to use real top model', )

    # print paras
    parser.add_argument('--print-to-txt', default=0, type=int, choices=[0, 1], help='save all outputs to txt or not')

    parser.add_argument('--gpu', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    args.resume_bottom = args.resume_dir + f'/saved_models/{args.dataset}_saved_models/surrogate_test/' + args.resume_name_bottom
    args.resume_surrogate = args.resume_dir + f'/saved_surrogate/{args.dataset}_saved_models/' + args.resume_name_surrogate

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_cuda = torch.cuda.is_available()
    
    main()