from common import *


class DiceLoss_ternaus:
    def __init__(self, dice_weight=1):
        self.nll_loss = nn.BCELoss()
        self.dice_weight = dice_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.dice_weight:
            eps = 1.
            dice_target = (targets == 1).float()
            dice_output = outputs
            intersection = (dice_output * dice_target).sum() + eps
            union = dice_output.sum() + dice_target.sum() + eps

            loss -= torch.log(2 * intersection / union)

        return loss


class IOULoss:
    def __init__(self, dice_weight=1):
        self.dice_weight = dice_weight

    def __call__(self, outputs, targets):
        if self.dice_weight:
            dice_target = (targets == 1).float()
            dice_output = outputs
            intersection = (dice_output * dice_target).sum()
            union = dice_output.sum() + dice_target.sum()

            loss = torch.log(intersection / (union-intersection))

        return loss

def dice_coef_loss(input, target):
    smooth = 1e-7

    intersection = -2. * ((target * input).sum() + smooth)
    union = target.sum() + input.sum() + smooth
    # print ("intersection and union:", intersection, union)
    return (intersection / union)


def dice_coef_loss_multi(input, target, numclass, print_score=False):
    smooth = 1e-7
    score = 0
    for idx in range(numclass):

        # print ('target sum = ',target[:,idx+1,:,:].sum())
        # print ('input sum = ',input[:,idx+1,:,:].sum())

        intersection = -2. * ((target[:,idx+1,:,:] * input[:,idx+1,:,:]).sum() + smooth)
        union = target[:,idx+1,:,:].sum() + input[:,idx+1,:,:].sum() + smooth
        score += (intersection / union) 
        if (print_score==True):
            print ('Score for class %d : %f \n' % (idx+1, intersection / union))
    return (score/numclass)

def bce_dice_loss(input, target):
    dicescore = dice_coef_loss(input, target)
    bcescore = nn.functional.binary_cross_entropy(input, target)


    return bcescore + (1. + dicescore)