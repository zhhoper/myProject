import pandas as pd
from torch.autograd import Variable
import numpy as np
import torch

class evaluateNetwork(object):
    '''
        compute loss
    '''
    def __init__(self, criterion_mask, criterion_WHDR, criterion_SAW_1, criterion_SAW_2, gradientLayer): 
        # we may use three criterions:
        # 1. mask loss (including SiMSE loss)
        # 2. WHDR loss (loss for albedo)
        # 3. SAW loss (loss for shading)
        self.criterion_mask = criterion_mask
        self.criterion_WHDR = criterion_WHDR
        self.criterion_SAW_1 = criterion_SAW_1
        self.criterion_SAW_2 = criterion_SAW_2
        self.gradientLayer = gradientLayer
        self.eps = 1e-6
        self.SUNCG_list = ['albedo', 'normal', 'shading']
        self.numCompare = 300 # maximum number of comparison for training and testing

    def getLoss_ABS(self, source, target, mask):
        '''
            return the mask loss with abs
        '''
        loss = self.criterion_mask(mask, torch.abs(source - target))
        return loss

    def getLoss_L2(self, source, target):
        '''
            return the l2 loss
            for reconstruction
        '''
        loss = torch.sqrt(torch.mean((source - target)**2))
        return loss

    def getLoss_WHDR(self, source, label):
        '''
            get WHDR loss
        '''
        loss = self.criterion_WHDR(source, label)
        return loss

    def getLoss_SAW_1(self, source, image, mask, numLabel):
        '''
            get SAW loss 1, loss for boundary of shadow
        '''
        loss = self.criterion_SAW_1(source, image, mask, numLabel)
        return loss

    def getLoss_SAW_2(self, source, mask, numLabel):
        '''
            get SAW loss 2, loss for smooth region of shading
        '''
        loss = self.criterion_SAW_2(source, mask, numLabel)
        return loss


    def getLoss_lighting(self, output_lighting):
        pass

    def getAlpha(self, source, target):
        '''
            compute a alpha to minimize
            (alpha*source - target)^2 
        '''
        if source.size()[0] != target.size()[0] or \
        	source.size()[1] != target.size()[1] or \
        	source.size()[2] != target.size()[2] or \
        	source.size()[3] != target.size()[3]:
                raise ValueError('size of groud truth and ouptut does not match')
        numImages = source.shape[0]
        source = source.view(numImages, -1)
        target = source.view(numImages, -1)
                    
        alpha = torch.sum(target*source,dim=1)/(torch.sum(source**2, dim=1) + self.eps)
        return alpha

    def loadLabel(self, labelName, dataName):
        '''
            load csv file 
        '''
        data_csv = pd.read_csv(labelName).values
        data_csv = data_csv[:,2:]

        if dataName == 'SUNCG':
            # for suncg dataset, select 3000 comparisons
            tmpLabel = data_csv[:,-2]
            ind_equal = np.where(tmpLabel == 0)[0]
            ind_darker = np.where(tmpLabel == 1)[0]
            ind_lighter = np.where(tmpLabel == 2)[0]
            if ind_equal.shape[0] > self.numCompare:
                np.random.shuffle(ind_equal)
                ind_equal = ind_equal[0:self.numCompare]
            if ind_darker.shape[0] > self.numCompare:
                np.random.shuffle(ind_darker)
                ind_equal = ind_equal[0:self.numCompare]
            if ind_lighter.shape[0] > self.numCompare:
                np.random.shuffle(ind_lighter)
                ind_lighter = ind_equal[0:self.numCompare]
            data_csv = np.concatenate((data_csv[ind_equal], data_csv[ind_darker],
                data_csv[ind_lighter]), axis=0)

        return data_csv

    def IIW_loss(self, sourceData, targetData, dataName):
        '''
            compute IIW loss
        '''
        loss = {}
        # loss of WHDR
        WHDR_labelName = targetData['label']
        WHDR_loss = Variable(torch.cuda.FloatTensor([0]))
        # load label and compute the loss
        for (i, labelName) in enumerate(WHDR_labelName):
            label = Variable(torch.from_numpy(self.loadLabel(labelName, dataName)).cuda(), requires_grad=True).float()
            WHDR_loss += self.getLoss_WHDR(sourceData['albedo'][i], label)
        loss['WHDR'] = WHDR_loss/len(WHDR_labelName)
        return loss
        
    def SUNCG_data(self, sourceData, targetData):
        '''
            SUNCG data contains:
            mask loss for albedo, shading, normal
            WHDR loss for albedo
        '''
        # use l1 loss
        mask = targetData['mask']
        imgMask = mask.expand(-1, 3, -1, -1)
        gradMask = mask.expand(-1, 6, -1, -1)
        loss = {}
        for item in self.SUNCG_list:
            source = sourceData[item]
            target = targetData[item]
            loss[item] = self.getLoss_ABS(source, target, imgMask)
            source_grad = self.gradientLayer.forward(source)
            target_grad = self.gradientLayer.forward(target)
            loss[item + '_grad'] = self.getLoss_ABS(source_grad, target_grad, gradMask)

        ## loss of WHDR
        #WHDR_labelName = targetData['label']
        #WHDR_loss = 0
        ## load label and compute the loss
        #for (i, labelName) in enumerate(WHDR_labelName):
        #    label = Variable(self.loadLabel(labelName, 'SUNCG').cuda()).float()
        #    WHDR_loss += self.getLoss_WHDR(source['albedo'][i], label)
        #loss['WHDR'] = WHDR_loss/len(WHDR_labelName)
        return  loss


    def IIW_data(self, sourceData, images, targetData):
        '''
            IIW data contains:
            WHDR loss for albedo
            SAW loss for shading
        '''
        loss = {}
        # WHDR loss
        #WHDR_labelName = targetData['label']
        #WHDR_loss = Variable(torch.cuda.FloatTensor(1))
        #WHDR_loss[0] = 0
        #for (i, labelName) in enumerate(WHDR_labelName):
        #    label = Variable(self.loadLabel(labelName, 'IIW').cuda()).float()
        #    WHDR_loss += self.getLoss_WHDR(source['albedo'][i], label)

        # SAW loss
        saw_loss_1 = Variable(torch.cuda.FloatTensor([0]))
        saw_loss_2 = Variable(torch.cuda.FloatTensor([0]))
        total_num = sourceData['shading'].shape[0]
        for i in range(total_num):
            shading = sourceData['shading'][i]
            saw_loss_1 += self.getLoss_SAW_1(shading, images, targetData['saw_mask_1'][i], 
                    targetData['num_saw_mask_1'][i])
            saw_loss_2 += self.getLoss_SAW_2(shading, targetData['saw_mask_2'][i],
                    targetData['num_saw_mask_2'][i])
        loss['saw_loss_1'] = saw_loss_1/total_num
        loss['saw_loss_2'] = saw_loss_2/total_num

        # reconstruction loss
        reconstruct_img = sourceData['albedo']*sourceData['shading']
        alpha = self.getAlpha(reconstruct_img, images)
 
        # ugly implementation
        numImages, numChannels, numRows, numCols = reconstruct_img.shape
        tmp_alpha = alpha.view(numImages, 1).repeat(1, numChannels, 
                numRows, numCols)
        tmp_alpha_1 = tmp_alpha.view(numImages, numChannels, numRows, numCols)
        alphaImages = tmp_alpha_1*reconstruct_img
        loss['reconstruct'] = self.getLoss_L2(alphaImages, images)
        return loss

    def NYU_data(self, sourceData, images, targetData):
        '''
            NYU data contains:
            SAW loss for shading
            mask loss for normal
        '''
        loss = {}
        mask = targetData['mask']
        # loss for normal
        imgMask = mask.expand(-1, 3, -1, -1)
        gradMask = mask.expand(-1, 6, -1, -1)
        source = sourceData['normal']
        target = targetData['normal']
        loss['normal'] = self.getLoss_ABS(source, target, imgMask)
        source_grad = self.gradientLayer.forward(source)
        target_grad = self.gradientLayer.forward(target)
        loss['normal_grad'] = self.getLoss_ABS(source_grad, target_grad, gradMask)

        # SAW loss
        saw_loss_1 = Variable(torch.cuda.FloatTensor([0]))
        saw_loss_2 = Variable(torch.cuda.FloatTensor([0]))
        total_num = sourceData['shading'].shape[0]
        for i in range(total_num):
            shading = sourceData['shading'][i]
            saw_loss_1 += self.getLoss_SAW_1(shading, images, targetData['saw_mask_1'][i], 
                    targetData['num_saw_mask_1'][i])
            saw_loss_2 += self.getLoss_SAW_2(shading, targetData['saw_mask_2'][i],
                    targetData['num_saw_mask_2'][i])
        loss['saw_loss_1'] = saw_loss_1/total_num
        loss['saw_loss_2'] = saw_loss_2/total_num

        # reconstruction loss
        reconstruct_img = sourceData['albedo']*sourceData['shading']
        alpha = self.getAlpha(reconstruct_img, images)

        # ugly implementation
        numImages, numChannels, numRows, numCols = reconstruct_img.shape
        tmp_alpha = alpha.view(numImages, 1).repeat(1, numChannels, 
                numRows, numCols)
        tmp_alpha_1 = tmp_alpha.view(numImages, numChannels, numRows, numCols)
        alphaImages = tmp_alpha_1*reconstruct_img
        loss['reconstruct'] = self.getLoss_L2(alphaImages, images)
        return loss
