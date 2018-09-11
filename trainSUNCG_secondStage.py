import sys
sys.path.append('utils')
sys.path.append('model')

import os
import numpy as np
import time

import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# load module defined by myself
from defineHourglass_16 import *
from defineNetworks_overlapping import *
from utils_shading import *
from defineCriterion import *
from loadData_basic import *
from defineHelp import *
from defineHelp_lighting import *

# set random seed for all possible randomness
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# image size to deal with
IMAGE_SIZE=120

baseFolder = '/scratch1/intrinsicImage/synthetic_SUNCG/'

criterion = mask_loss()

my_network = HourglassNet(27)
my_network.cuda()
gradientLayer = gradientLayer_color()
gradientLayer.cuda()
shadingLayer = constructShading()
shadingLayer.cuda()
samplingLightLayer = samplingLight()
samplingLightLayer.cuda()
my_convNet = myConv_subNetwork(16, 16, 8, 8, IMAGE_SIZE, IMAGE_SIZE)
my_convNet.cuda()


transformer = transforms.Compose([cropImg(output_size=IMAGE_SIZE), ToTensor()])
transformer_test = transforms.Compose([testTransfer(output_size=IMAGE_SIZE), ToTensor()])

trainLoaderHelper = SUNCG(dataFolder = os.path.join(baseFolder, 'images_color'), 
    albedoFolder=os.path.join(baseFolder, 'albedo'),
    shadingFolder=os.path.join(baseFolder, 'shading'),
    normalFolder='/scratch1/data/SUNCG_groundTruth/SUNCG_normal',
    maskFolder=os.path.join(baseFolder, 'images_mask_correct'),
    fileListName=os.path.join(baseFolder, 'training.list'),
    transform = transformer)

testLoaderHelper = SUNCG(dataFolder=os.path.join(baseFolder, 'images_color'),
    albedoFolder=os.path.join(baseFolder, 'albedo'),
    shadingFolder=os.path.join(baseFolder, 'shading'),
    normalFolder='/scratch1/data/SUNCG_groundTruth/SUNCG_normal',
    maskFolder=os.path.join(baseFolder, 'images_mask_correct'),
    fileListName=os.path.join(baseFolder, 'testing.list'),
    transform = transformer_test)

trainLoader = torch.utils.data.DataLoader(trainLoaderHelper, 
    batch_size=10, shuffle=True, num_workers=5)

testLoader = torch.utils.data.DataLoader(testLoaderHelper, 
    batch_size=10, shuffle=False, num_workers=5)

# get the normal/direction of positive and negative spheres
# used to constrain lighting to be positive
sphereDirection_pos, sphereDirection_neg, sphereMask = getSHNormal(64)
sphereDirection_pos = Variable(torch.from_numpy(sphereDirection_pos).cuda()).float()
sphereDirection_neg = Variable(torch.from_numpy(sphereDirection_neg).cuda()).float()
sphereMask = Variable(torch.from_numpy(sphereMask).cuda()).float()

def getLoss(output_albedo, output_shading, output_normal, albedo, shading, normal, mask):
    '''
        compute the loss for the mini-batch
    '''
    output_albedo_grad = gradientLayer.forward(output_albedo)
    albedo_grad = gradientLayer.forward(albedo)
    output_shading_grad = gradientLayer.forward(output_shading)
    shading_grad = gradientLayer.forward(shading)
    output_normal_grad = gradientLayer.forward(output_normal)
    normal_grad = gradientLayer.forward(normal)
    
    imgMask = mask.expand(-1, 3, -1, -1)
    gradMask = mask.expand(-1, 6, -1, -1)
    
    loss_albedo = criterion(imgMask, torch.abs(albedo - output_albedo))
    loss_shading = criterion(imgMask, torch.abs(shading - output_shading))
    
    loss_albedo_grad = criterion(gradMask, torch.abs(albedo_grad - output_albedo_grad))
    loss_shading_grad = criterion(gradMask, torch.abs(shading_grad - output_shading_grad))
    
    #loss_normal = criterion(mask, torch.sum(-1*output_normal*normal, dim=1, keepdim=True))
    loss_normal = criterion(imgMask, torch.abs(output_normal - normal))
    loss_normal_grad = criterion(gradMask, torch.abs(normal_grad - output_normal_grad))
    
    return loss_albedo, loss_shading, loss_albedo_grad, loss_shading_grad, loss_normal, loss_normal_grad

def networkForward(data, Testing=False):
    '''
        given data, compute the loss for the network
        return loss in a list
    '''
    inputs, albedo, shading, normal, mask = data
    inputs, albedo, shading, normal, mask = \
            Variable(inputs.cuda(), volatile=Testing).float(), \
            Variable(albedo.cuda(), volatile=Testing).float(), \
            Variable(shading.cuda(), volatile=Testing).float(), \
            Variable(normal.cuda(), volatile=Testing).float(), \
            Variable(mask.cuda(), volatile=Testing).float()

    # --------------------------------------------------------------------------------
    # get albedo, shading, normal, lighting in coarse scale and prepare the residual
    coarse_albedo, coarse_normal, coarse_lighting = \
            my_network_coarse(F.upsample(inputs, size=[64, 64], mode='bilinear'))
    coarse_shading = shadingLayer(coarse_normal, coarse_lighting)
    coarse_albedo = F.upsample(coarse_albedo, size=[IMAGE_SIZE, IMAGE_SIZE], mode='bilinear')
    coarse_normal = F.upsample(coarse_normal, size=[IMAGE_SIZE, IMAGE_SIZE], mode='bilinear')
    coarse_shading = F.upsample(coarse_shading, size=[IMAGE_SIZE, IMAGE_SIZE], mode='bilinear')

    # NOTE: we have a bug in coarse network for lighting, correct it
    coarse_lighting = coarse_lighting[:,0:27]
    
    # get the residual as groundtruth
    albedo = albedo - coarse_albedo
    shading = shading - coarse_shading
    normal = normal - coarse_normal
    
    # concatenate images, albedo, normal, shading as input
    inputs = torch.cat((inputs, coarse_albedo, coarse_normal, coarse_shading), dim=1)
    # predict residual
    output_albedo, output_normal, output_shading, output_lighting, overlap_loss = \
        my_convNet(inputs, coarse_normal, my_network)

    # compute loss
    loss_albedo, loss_shading, loss_albedo_grad, loss_shading_grad, \
        loss_normal, loss_normal_grad = getLoss(output_albedo, 
        output_shading, output_normal, albedo, shading, normal, mask)
    
    # loss for lighting
    loss_light_pos = nonNegativeLighting_fineLevel(samplingLightLayer, 
            sphereDirection_pos, output_lighting, coarse_lighting)
    loss_light_neg = nonNegativeLighting_fineLevel(samplingLightLayer, 
            sphereDirection_neg, output_lighting, coarse_lighting)
    loss_light = loss_light_neg + loss_light_pos

    # append all the losses
    loss = {}
    loss['albedo'] = loss_albedo
    loss['albedo_grad'] = loss_albedo_grad
    loss['shading'] = loss_shading
    loss['shading_grad'] = loss_shading_grad
    loss['normal'] = loss_normal
    loss['normal_grad'] = loss_normal_grad
    loss['light'] = loss_light
    loss['overlap'] = overlap_loss
    loss['loss_list'] = [loss_albedo.data[0], loss_albedo_grad.data[0], loss_shading.data[0],
        loss_shading_grad.data[0], loss_normal.data[0], loss_normal_grad.data[0], loss_light.data[0],
        overlap_loss.data[0]]

    return loss

def main(savePath, load_path, load_model, lr=1e-3, weight_decay=0, total_epoch=100):
    begin_time = time.time()
    print 'learning rate is %.6f' % lr
    print 'weight decay is %.6f' % weight_decay
    print 'epoch is %05d' % total_epoch
    savePath = savePath + '_{:0.4f}_{:0.2f}_{:04d}'.format(lr, weight_decay, total_epoch)
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    saveIntermedia = os.path.join(savePath, 'trained_model')
    if not os.path.exists(saveIntermedia):
        os.makedirs(saveIntermedia)
    writer = SummaryWriter(os.path.join(savePath, 'tensorboard'))
    
    global my_network_coarse 
    my_network_coarse = torch.load(os.path.join(load_path, 'trained_model.t7'))
    my_network_coarse.cuda()

    my_network.load_state_dict(torch.load(load_model))
    
    monitor_count = 20
    optimizer = optim.Adam(my_network.parameters(), lr = lr, weight_decay=weight_decay)
    fid = open(os.path.join(savePath, 'training.log'), 'w')
    fid_sep = open(os.path.join(savePath, 'training_sep.log'), 'w')
    fid_test = open(os.path.join(savePath, 'testing.log'), 'w')
    fid_test_sep = open(os.path.join(savePath, 'testing_sep.log'), 'w')
    print>>fid_sep, 'albedo, albedo_grad, shading, shading_grad, normal, normal_grad, lighting, overlap'
    print>>fid_test_sep, 'albedo, albedo_grad, shading, shading_grad, normal, normal_grad, lighting, overlap'

    numIte_training = 0
    numIte_testing = 0
    for epoch in range(total_epoch):
        # ---------------------------------------------------------
        # training
        # ---------------------------------------------------------
        running_loss = 0.0
        # loss for each indivisual component
        running_albedo_loss = 0.0
        running_shading_loss = 0.0
        running_normal_loss = 0.0
        running_light_loss = 0.0
        running_overlap_loss = 0.0
        
        my_network.train(True)
    	epoch_time = time.time()
    	loss_list = []
        for i , data in enumerate(trainLoader, 0):
            begin_time = time.time()
            optimizer.zero_grad()
            loss_miniBatch = networkForward(data, Testing=False)
            loss = loss_miniBatch['albedo'] + loss_miniBatch['albedo_grad'] + \
                    loss_miniBatch['shading'] + loss_miniBatch['shading_grad'] + \
                    loss_miniBatch['normal'] + loss_miniBatch['normal_grad'] + \
                    loss_miniBatch['light'] + loss_miniBatch['overlap']*0.25

            loss.backward()

            optimizer.step()

            loss_list.append(loss_miniBatch['loss_list'])

            running_loss += loss.data[0]
            running_albedo_loss  += loss_miniBatch['albedo'].data[0]
            running_shading_loss  += loss_miniBatch['shading'].data[0]
            running_normal_loss  += loss_miniBatch['normal'].data[0]
            running_light_loss  += loss_miniBatch['light'].data[0]
            running_overlap_loss  += loss_miniBatch['overlap'].data[0]


            # record loss
            if i%monitor_count == monitor_count - 1:
                numIte_training = numIte_training + 1
                print '[%d %5d] loss: ' % (epoch + 1, i+1),
                print '%.4f '*8 % tuple([item/monitor_count for item in loss_miniBatch['loss_list']])
                print>>fid, '%d %5d ' % (epoch+1, i+1),
                print>>fid, '%.4f '*8 % tuple([item/monitor_count for item in loss_miniBatch['loss_list']])

                writer.add_scalar('train/loss_albedo', running_albedo_loss/monitor_count, numIte_training)
                writer.add_scalar('train/loss_shading', running_shading_loss/monitor_count, numIte_training)
                writer.add_scalar('train/loss_normal', running_normal_loss/monitor_count, numIte_training)
                writer.add_scalar('train/loss_lighting', running_light_loss/monitor_count, numIte_training)
                writer.add_scalar('train/loss_overlap', running_overlap_loss/monitor_count, numIte_training)

                running_loss = 0
                running_albedo_loss = 0
                running_shading_loss = 0
                running_normal_loss = 0
                running_light_loss = 0
                running_overlap_loss = 0

            #print 'One iteration is %s second' % (time.time() - begin_time)
        loss_list = np.mean(np.array(loss_list), axis=0)
        print >> fid_sep, '%0.6f '* 8 % tuple(loss_list)
        print '%0.6f '* 8 % tuple(loss_list)
        # ---------------------------------------------------------
        # validation 
        # ---------------------------------------------------------
        my_network.train(False)
        test_loss = 0
        # loss for each component
        test_albedo_loss = 0
        test_shading_loss = 0
        test_normal_loss = 0
        test_light_loss = 0
        test_overlap_loss = 0

        count = 0
        test_loss_list = []
        for i, data in enumerate(testLoader, 0):
            loss_miniBatch = networkForward(data, Testing=True)
            loss = loss_miniBatch['albedo'] + loss_miniBatch['albedo_grad'] + \
                    loss_miniBatch['shading'] + loss_miniBatch['shading_grad'] + \
                    loss_miniBatch['normal'] + loss_miniBatch['normal_grad'] + \
                    loss_miniBatch['light'] + loss_miniBatch['overlap']*0.25
            test_loss_list.append(loss_miniBatch['loss_list'])
            test_loss += loss.data[0]
            test_loss += loss.data[0]
            test_albedo_loss  += loss_miniBatch['albedo'].data[0]
            test_shading_loss  += loss_miniBatch['shading'].data[0]
            test_normal_loss  += loss_miniBatch['normal'].data[0]
            test_light_loss  += loss_miniBatch['light'].data[0]
            test_overlap_loss  += loss_miniBatch['overlap'].data[0]
        
            count = count + 1
        print '[%d  ] test loss: ' % (epoch+1),
        print '%.4f '*8 % tuple([item/count for item in loss_miniBatch['loss_list']])
        print>>fid_test, '%d ' % (epoch+1),
        print>>fid_test, '%.4f '*8 % tuple([item/count for item in loss_miniBatch['loss_list']])
        
        writer.add_scalar('test/loss_albedo', test_albedo_loss/count, epoch)
        writer.add_scalar('test/loss_shading', test_shading_loss/count, epoch)
        writer.add_scalar('test/loss_normal', test_normal_loss/count, epoch)
        writer.add_scalar('test/loss_light', test_light_loss/count, epoch)
        writer.add_scalar('test/loss_overlap', test_overlap_loss/count, epoch)
    
        test_loss_list = np.mean(np.array(test_loss_list), axis=0)
        print >> fid_test_sep, '%0.6f '* 8 % tuple(test_loss_list)
        print '%0.6f '* 8 % tuple(test_loss_list)

        tmp_saveName = os.path.join(saveIntermedia, 'trained_model_{:02d}.t7'.format(epoch))
        my_network.cpu()
        torch.save(my_network.state_dict(), tmp_saveName)
        my_network.cuda()
        print 'this epoch cost %s seconds' %(time.time() - epoch_time)

    my_network.cpu()
    torch.save(my_network, os.path.join(savePath, 'trained_model.t7'))
    print 'time used for training is %s' % (time.time() - begin_time)
    print('Finished training')
    
    fid.close()
    fid_test.close()
    fid_sep.close()
    fid_test_sep.close()

if __name__ == '__main__':
    savePath = sys.argv[1]
    if len(sys.argv) > 2:
        lr = float(sys.argv[2])
        weight_decay = float(sys.argv[3])
        total_epoch=int(sys.argv[4])
        load_path = sys.argv[5]
        load_model = sys.argv[6]
        main(savePath, load_path, load_model, lr, weight_decay, total_epoch)
    else:
        main(savePath)
