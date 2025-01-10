"""
Thanks to Yerlan Idelbayev.
"""
import torch.nn as nn
import torch.nn.functional as F
from my_utils.utils import weights_init,  BasicBlock
import torch

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, kernel_size, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], kernel_size, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], kernel_size, stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, kernel_size, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # [bs,3,32,16]
        out = F.relu(self.bn1(self.conv1(x)))
        # [bs,16,32,16]
        out = self.layer1(out)
        # [bs,16,32,16]
        out = self.layer2(out)
        # [bs,32,16,8]
        out = self.layer3(out)
        # [bs,64,8,4]
        out = F.avg_pool2d(out, out.size()[2:])
        # [bs,64,1,1]
        out = out.view(out.size(0), -1)
        # [bs,64]
        out = self.linear(out)
        # [bs,10]
        return out


def resnet20(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[3, 3, 3], kernel_size=kernel_size, num_classes=num_classes)


def resnet110(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[18, 18, 18], kernel_size=kernel_size, num_classes=num_classes)


def resnet56(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[9, 9, 9], kernel_size=kernel_size, num_classes=num_classes)


# CIFAR10-----------------------------------------------------------------------------------
# original top    
class TopModelForCifar10(nn.Module):
    def __init__(self):
        super(TopModelForCifar10, self).__init__()
        self.fc1top = nn.Linear(20, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)

        self.apply(weights_init)
    
    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        # 沿着维度1拼接输入张量，接收来自两个不同模型的输入张量，将它们拼接在一起。
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models

        # 通过带有批归一化和ReLU激活的层进行前向传播
        x = self.fc1top(F.relu(self.bn0top(x)))  #  应用第一个线性层，然后是批归一化和ReLU激活
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))

        # 应用Log Softmax以获得最终输出
        return F.log_softmax(x, dim=1)

# original bottom
class BottomModelForCifar10(nn.Module):  # BottomModelForCifar10 类继承自 nn.Module
    def __init__(self):
        super(BottomModelForCifar10, self).__init__()  # 调用 nn.Module 类的构造函数
        self.resnet20 = resnet20(num_classes=10)  # 创建一个 resnet20 模型，输出类别数为 10

    def forward(self, x):
        x = self.resnet20(x)  # 模型前向传播：输入 x 经过底部模型的 ResNet-20 结构
        return x

# 对应原始top的代理模型结构。代理模型和真top的区别就是输入是拼接过的
class SurrogateModelForCifar10(nn.Module):
    def __init__(self):
        super(SurrogateModelForCifar10, self).__init__()
        self.fc1top = nn.Linear(20, 20)
        self.fc2top = nn.Linear(20, 10)
        self.fc3top = nn.Linear(10, 10)
        self.fc4top = nn.Linear(10, 10)
        # self.fc5top = nn.Linear(10, 10)
        self.bn0top = nn.BatchNorm1d(20)
        self.bn1top = nn.BatchNorm1d(20)
        self.bn2top = nn.BatchNorm1d(10)
        self.bn3top = nn.BatchNorm1d(10)
        # self.bn4top = nn.BatchNorm1d(10)

        self.apply(weights_init)
    
    def forward(self, output_bottom_models):
        # 沿着维度1拼接输入张量，接收来自两个不同模型的输入张量，将它们拼接在一起。
        x = output_bottom_models

        # 通过带有批归一化和ReLU激活的层进行前向传播
        x = self.fc1top(F.relu(self.bn0top(x)))  #  应用第一个线性层，然后是批归一化和ReLU激活
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        # x = self.fc5top(F.relu(self.bn4top(x)))

        # 应用Log Softmax以获得最终输出
        return F.log_softmax(x, dim=1)

# CIFAR100-------------------------------------------------------------------------
# original top and bottom model
class BottomModelForCifar100(nn.Module):
    def __init__(self):
        super(BottomModelForCifar100, self).__init__()
        self.resnet20 = resnet20(num_classes=100)

    def forward(self, x):
        x = self.resnet20(x)
        return x

class TopModelForCifar100(nn.Module):
    def __init__(self):
        super(TopModelForCifar100, self).__init__()
        self.fc1top = nn.Linear(200, 200)
        self.fc2top = nn.Linear(200, 100)
        self.fc3top = nn.Linear(100, 100)
        self.fc4top = nn.Linear(100, 100)
        self.bn0top = nn.BatchNorm1d(200)
        self.bn1top = nn.BatchNorm1d(200)
        self.bn2top = nn.BatchNorm1d(100)
        self.bn3top = nn.BatchNorm1d(100)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)

# 对应原始top的代理模型结构。代理模型和真top的区别就是输入是拼接过的
class SurrogateModelForCifar100(nn.Module):
    def __init__(self):
        super(SurrogateModelForCifar100, self).__init__()
        self.fc1top = nn.Linear(200, 200)
        self.fc2top = nn.Linear(200, 100)
        self.fc3top = nn.Linear(100, 100)
        self.fc4top = nn.Linear(100, 100)
        self.bn0top = nn.BatchNorm1d(200)
        self.bn1top = nn.BatchNorm1d(200)
        self.bn2top = nn.BatchNorm1d(100)
        self.bn3top = nn.BatchNorm1d(100)

        self.apply(weights_init)
    
    def forward(self, output_bottom_models):
        # 沿着维度1拼接输入张量，接收来自两个不同模型的输入张量，将它们拼接在一起。
        x = output_bottom_models

        # 通过带有批归一化和ReLU激活的层进行前向传播
        x = self.fc1top(F.relu(self.bn0top(x)))  #  应用第一个线性层，然后是批归一化和ReLU激活
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))

        # 应用Log Softmax以获得最终输出
        return F.log_softmax(x, dim=1)

# TinyImageNet----------------------------------------------------------------------
# original top and bottom model
class BottomModelForTinyImageNet(nn.Module):
    def __init__(self):
        super(BottomModelForTinyImageNet, self).__init__()
        self.resnet56 = resnet56(num_classes=200)

    def forward(self, x):
        x = self.resnet56(x)
        return x

class TopModelForTinyImageNet(nn.Module):
    def __init__(self):
        super(TopModelForTinyImageNet, self).__init__()
        self.fc1top = nn.Linear(400, 400)
        self.fc2top = nn.Linear(400, 200)
        self.fc3top = nn.Linear(200, 200)
        self.bn0top = nn.BatchNorm1d(400)
        self.bn1top = nn.BatchNorm1d(400)
        self.bn2top = nn.BatchNorm1d(200)

        self.apply(weights_init)

    def forward(self, input_tensor_top_model_a, input_tensor_top_model_b):
        output_bottom_models = torch.cat((input_tensor_top_model_a, input_tensor_top_model_b), dim=1)
        x = output_bottom_models
        x = self.fc1top(F.relu(self.bn0top(x)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        return F.log_softmax(x, dim=1)

class SurrogateModelForTinyImageNet(nn.Module):
    def __init__(self):
        super(SurrogateModelForTinyImageNet, self).__init__()
        self.fc1top = nn.Linear(400, 400)
        self.fc2top = nn.Linear(400, 200)
        self.fc3top = nn.Linear(200, 200)
        self.bn0top = nn.BatchNorm1d(400)
        self.bn1top = nn.BatchNorm1d(400)
        self.bn2top = nn.BatchNorm1d(200)

        self.apply(weights_init)
    
    def forward(self, output_bottom_models):
        # 沿着维度1拼接输入张量，接收来自两个不同模型的输入张量，将它们拼接在一起。
        x = output_bottom_models

        # 通过带有批归一化和ReLU激活的层进行前向传播
        x = self.fc1top(F.relu(self.bn0top(x)))  #  应用第一个线性层，然后是批归一化和ReLU激活
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))

        # 应用Log Softmax以获得最终输出
        return F.log_softmax(x, dim=1)


D_ = 2 ** 13


class BottomModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_model(self, half, is_adversary, n_labels=10):
        if self.dataset_name == 'CIFAR10':
            return BottomModelForCifar10()
        elif self.dataset_name == 'CIFAR100':
            return BottomModelForCifar100()
        elif self.dataset_name == 'TinyImageNet':
            return BottomModelForTinyImageNet()
        else:
            raise Exception('Unknown dataset name!')

    def __call__(self):
        raise NotImplementedError()


class TopModel:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def get_model(self, surrogate=False):
        if surrogate:
            if self.dataset_name == 'CIFAR10':
                return SurrogateModelForCifar10()
            elif self.dataset_name == 'CIFAR100':
                return SurrogateModelForCifar100()
            elif self.dataset_name == 'TinyImageNet':
                return SurrogateModelForTinyImageNet()
        else:
            if self.dataset_name == 'CIFAR10':
                return TopModelForCifar10()
            elif self.dataset_name == 'CIFAR100':
                return TopModelForCifar100()
            elif self.dataset_name == 'TinyImageNet':
                return TopModelForTinyImageNet()
            else:
                raise Exception('Unknown dataset name!')


def update_top_model_one_batch(optimizer, model, output, batch_target, loss_func):
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def update_bottom_model_one_batch(optimizer, model, output, batch_target, loss_func):
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return

def update_bottom_model_one_batch_pro(optimizer, model, output, batch_target, anchor, positives, negatives, loss_func):
    loss = loss_func(output, batch_target, anchor, positives, negatives)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return


if __name__ == "__main__":
    demo_model = BottomModel(dataset_name='CIFAR10')
    print(demo_model)
