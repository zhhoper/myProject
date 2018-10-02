from torch.autograd import Variable

class labelToGPU:
    '''
        transfer everything to gpu
    '''
    def __init__(self):
        self.item_list = ['albedo', 'shading', 'normal', 'mask', 
                'saw_mask_1', 'saw_mask_2', 'num_saw_mask_1', 
                'num_saw_mask_2']
    def toGPU(self, labelData):
        for item in labelData.keys():
            if item in self.item_list:
                labelData[item] = Variable(labelData[item].cuda()).float()
        return labelData


