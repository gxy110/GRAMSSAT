import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_sets import BottomModel, TopModel
# from liver_models import BottomModel
import torch.nn.init as init


def weights_init_ones(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.ones_(m.weight)


class BottomModelPlus(nn.Module):
    def __init__(self, size_bottom_out, num_classes, num_layer=1, activation_func_type='ReLU', use_bn=True):
        super(BottomModelPlus, self).__init__()
        self.bottom_model = BottomModel(dataset_name=None)

        dict_activation_func_type = {'ReLU': F.relu, 'Sigmoid': F.sigmoid, 'None': None}
        self.activation_func = dict_activation_func_type[activation_func_type]
        self.num_layer = num_layer
        self.use_bn = use_bn

        self.fc_1 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_1 = nn.BatchNorm1d(size_bottom_out)
        self.fc_1.apply(weights_init_ones)

        self.fc_2 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_2 = nn.BatchNorm1d(size_bottom_out)
        self.fc_2.apply(weights_init_ones)

        self.fc_3 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_3 = nn.BatchNorm1d(size_bottom_out)
        self.fc_3.apply(weights_init_ones)

        self.fc_4 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_4 = nn.BatchNorm1d(size_bottom_out)
        self.fc_4.apply(weights_init_ones)

        self.fc_final = nn.Linear(size_bottom_out, num_classes, bias=True)
        self.bn_final = nn.BatchNorm1d(size_bottom_out)
        self.fc_final.apply(weights_init_ones)

    def forward(self, x):
        x = self.bottom_model(x)

        if self.num_layer >= 2:
            if self.use_bn:
                x = self.bn_1(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_1(x)

        if self.num_layer >= 3:
            if self.use_bn:
                x = self.bn_2(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_2(x)

        if self.num_layer >= 4:
            if self.use_bn:
                x = self.bn_3(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_3(x)

        if self.num_layer >= 5:
            if self.use_bn:
                x = self.bn_4(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_4(x)

        if self.use_bn:
            x = self.bn_final(x)
        if self.activation_func:
            x = self.activation_func(x)
        x = self.fc_final(x)

        return x
    

class SurrogateTopModel(nn.Module):
    def __init__(self, size_bottom_out, num_classes, num_layer=1, activation_func_type='ReLU', use_bn=True):
        super(SurrogateTopModel, self).__init__()

        dict_activation_func_type = {'ReLU': F.relu, 'Sigmoid': F.sigmoid, 'None': None}
        self.activation_func = dict_activation_func_type[activation_func_type]
        self.num_layer = num_layer
        self.use_bn = use_bn

        self.fc_1 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_1 = nn.BatchNorm1d(size_bottom_out)
        self.fc_1.apply(weights_init_ones)

        self.fc_2 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_2 = nn.BatchNorm1d(size_bottom_out)
        self.fc_2.apply(weights_init_ones)

        self.fc_3 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_3 = nn.BatchNorm1d(size_bottom_out)
        self.fc_3.apply(weights_init_ones)

        self.fc_4 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_4 = nn.BatchNorm1d(size_bottom_out)
        self.fc_4.apply(weights_init_ones)

        self.fc_final = nn.Linear(size_bottom_out, num_classes, bias=True)
        self.bn_final = nn.BatchNorm1d(size_bottom_out)
        self.fc_final.apply(weights_init_ones)

    def forward(self, x):
        if self.num_layer >= 2:
            if self.use_bn:
                x = self.bn_1(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_1(x)

        if self.num_layer >= 3:
            if self.use_bn:
                x = self.bn_2(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_2(x)

        if self.num_layer >= 4:
            if self.use_bn:
                x = self.bn_3(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_3(x)

        if self.num_layer >= 5:
            if self.use_bn:
                x = self.bn_4(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_4(x)

        if self.use_bn:
            x = self.bn_final(x)
        if self.activation_func:
            x = self.activation_func(x)
        x = self.fc_final(x)

        return x
    
class BottomAndSurrogateModel(nn.Module):
    def __init__(self, size_bottom_out, num_classes, num_layer=1, activation_func_type='ReLU', use_bn=True):
        super( BottomAndSurrogateModel, self).__init__()
        self.bottom_model_a = BottomModel(dataset_name=None)
        self.bottom_model_b = BottomModel(dataset_name=None)
        self.surrogate_model = SurrogateTopModel(size_bottom_out, num_classes, num_layer, activation_func_type, use_bn)
    
    def forward(self, x_a, x_b):
        x_a = self.bottom_model_a(x_a)
        x_b = self.bottom_model_b(x_b)
        x = torch.cat((x_a, x_b), dim=1)
        x = self.surrogate_model(x)
        return x

class BottomAndTopModel(nn.Module):
    def __init__(self, dataset_name):
        super( BottomAndTopModel, self).__init__()
        self.bottom_model_a = BottomModel(dataset_name=dataset_name)
        self.bottom_model_b = BottomModel(dataset_name=dataset_name)
        self.surrogate_model = TopModel(dataset_name=dataset_name).get_model(surrogate=True)
    
    def forward(self, x_a, x_b):
        x_a = self.bottom_model_a(x_a)
        x_b = self.bottom_model_b(x_b)
        x = torch.cat((x_a, x_b), dim=1)
        x = self.surrogate_model(x)
        return x