import torch

def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss

def cross_entropy_loss(x_pred, x_target):
    assert x_target.size() == x_pred.size(), "size fail ! "+str(x_target.size()) + " " + str(x_pred.size())
    
    logged_x_pred = torch.log(x_pred)
    cost_value = -torch.sum(x_target * logged_x_pred)
    
    return cost_value


def cross_entropy_cifar_loss(x_pred, x_target ):
    # x_pred.size(): [Batch size, num_classes]
    # x_target.size(): [Batch size]
    
    x_target_one_hot = torch.nn.functional.one_hot(x_target, num_classes=10)
    #print('x_pred', x_pred)
    
    loss_fn = torch.nn.MultiLabelSoftMarginLoss()
    
    return loss_fn(x_pred, x_target_one_hot)
