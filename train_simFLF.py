"""
create by xwj
"""
import sys
import os
import argparse
import torch.nn as nn
from src.simple_FLF import *
from lib.Myloader import *
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
import math
from src.SSIM import *
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from termcolor import cprint
from lib.lr import *
import time

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0.
    self.avg = 0.
    self.sum = 0.
    self.count = 0.

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def weights_init(model):
    if isinstance(model, list):
        for m in model:
            nn.init.kaiming_uniform_(m)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                #print torch.sum(m.weight)
                nn.init.kaiming_uniform_(m.weight)
                # m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)


def eva_model(escount, gtcount):
        mae = 0.
        mse = 0.
        for i in range(len(escount)):
            temp1 = abs(escount[i] - gtcount[i])
            temp2 = temp1 * temp1
            mae += temp1
            mse += temp2
        MAE = mae * 1. / len(escount)
        MSE = math.sqrt(1. / len(escount) * mse)
        return MAE, MSE

parser = argparse.ArgumentParser(description='PyTorch Triangle')

parser.add_argument('-train_im',default='/media/xwj/Data/DataSet/shanghai_tech/original/part_A_final/train_data/images',
                    help='path to train img')
parser.add_argument('-train_gt', default = '/media/xwj/Data/DataSet/shanghai_tech/original/part_A_final/train_data/ground_truth',
                    help='path to train gt')
parser.add_argument('-test_im', default='/media/xwj/Data/DataSet/shanghai_tech/original/part_A_final/test_data/images',
                    help='path to test img')
parser.add_argument('-test_gt', default = '/media/xwj/Data/DataSet/shanghai_tech/original/part_A_final/test_data/ground_truth',
                    help='path to test gt')

parser.add_argument('-resume',  default=True,type=str,
                    help='path to the pretrained model')

parser.add_argument('-multi-gpu',type=bool,default=True,
                    help='wheather to use multi gpu')

parser.add_argument('-lr', type=float,default=0.000001,
                    help='learning rate')

parser.add_argument('-method', type=str,default='Triangle-crop-fix-FA',
                    help='description of your method')

parser.add_argument('-epochs',type=int,default=1000,
                    help='how many epochs you wanna run')

parser.add_argument('-display',type=int,default=100,
                    help='how many epochs you wanna run')

parser.add_argument('-best_loss',default=float('inf'))
parser.add_argument('-best_mae',default=float('inf'))
parser.add_argument('-best_mse',default=float('inf'))
parser.add_argument('-start_epoch',type=int,default=0)


if __name__ == '__main__':
    global args
    args = parser.parse_args()

    current_dir = os.getcwd()
    saveimg = current_dir + '/data/' + args.method + '/img'
    savemodel = current_dir + '/data/' + args.method + '/model'
    savelog = current_dir + '/data/' + args.method + '/'
    ten_log = current_dir + '/runs/' + args.method

    need_dir = [saveimg, savemodel, savelog]
    for i in need_dir:
        if not os.path.exists(i):
            os.makedirs(i)
    writer = SummaryWriter(log_dir=ten_log)

    logger = logging.getLogger(name='train')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(savelog + 'output.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '[%(asctime)s]-[module:%(name)s]-[line: %(lineno)d]:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info('#' * 50)
    logger.info(args)

    train_data = SDNSHTech(imdir = args.train_im,gtdir=args.train_gt,transform= 0.5,train=True,test = False,raw = True,num_cut=4)
    val_data = SDNSHTech(imdir = args.test_im,gtdir=args.test_gt,train = False,test = True)

    train_loader = DataLoader(train_data,batch_size=1,shuffle=True,num_workers=8)
    val_loader = DataLoader(val_data,batch_size=1,shuffle=False,num_workers=0)

    net = Triangle(gray=False)

    # if args.multi_gpu:
    #     device = [0,1]
    #     net = torch.nn.DataParallel(net, device_ids=device)

    if args.resume:
        cprint('=> loading checkpoint ', color='yellow')
        checkpoint = torch.load(current_dir + '/data/Triangle-crop-fix/model/best_loss_mae.tar')
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        net.load_state_dict(checkpoint['state_dict'])
        lr = checkpoint['lr']
        cprint("=> loaded checkpoint ", color='yellow')
    else:
        weights_init(net)



    net.cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,momentum=0.9,weight_decay = 0.0005)
    scheduler = ReduceLROnPlateau(optimizer, 'min',verbose=True)
    LOSS = Myloss()
    logger.info(args.method)

    for epoch in range(args.start_epoch,args.epochs):
        epoch_start = time.time()
        trainmae = 0.
        trainmse = 0.
        valmae = 0.
        valmse = 0.
        trainloss = AverageMeter()
        valloss = AverageMeter()
        rightnow_lr = get_learning_rate(optimizer)
        logger.info('epoch:{} -- lr:{}'.format(epoch,rightnow_lr))

        escount = []
        gtcount = []
        ###############
        ####train start####

        net.train()
        train_start = time.time()

        for index,(img,den) in tqdm(enumerate(train_loader)):

            # img = img[0]
            # den = den[0]
            img = img.cuda()
            den = den.cuda()

            es_den = net(img)
            Loss = LOSS(es_den, den)
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            trainloss.update(Loss.item(), img.shape[0])

            es_count = np.sum(es_den[0][0].cpu().detach().numpy())
            gt_count = np.sum(den[0][0].cpu().detach().numpy())
            escount.append(es_count)
            gtcount.append(gt_count)
            # if index %50 ==0:
            #     cprint('MAE LOSS:%.8f - SSIM LOSS:%.8f' % (LOSS.loss_E, LOSS.loss_C), color='yellow')
        duration = time.time()-train_start
        trainfps = index/duration
        trainmae, trainmse = eva_model(escount, gtcount)

        info = 'trianloss :{%.8f} @ trainmae:{%.3f} @ trainmse:{%.3f} @ fps:{%.3f}' % (
            trainloss.avg ,
            trainmae, trainmse,
            trainfps)

        logger.info(info)

        del escount[:]
        del gtcount[:]

        ##############
        ####val start####

        valstart = time.time()
        net.eval()
        with torch.no_grad():

            for index,(vimg ,vden) in tqdm(enumerate(val_loader)):
                vimg = vimg.cuda()
                vden = vden.cuda()
                val_es_den = net(vimg)
                val_loss = LOSS(val_es_den,vden)
                valloss.update(val_loss.item(), vimg.shape[0])

                ves_count = np.sum(val_es_den[0][0].cpu().detach().numpy())
                vgt_count = np.sum(vden[0][0].cpu().detach().numpy())

                escount.append(ves_count)
                gtcount.append(vgt_count)



                if index % 60 ==0 and epoch % args.display == 0 :
                    cprint('MAE LOSS:%.8f - SSIM LOSS:%.8f'%(LOSS.loss_E,LOSS.loss_C),color='yellow')

                    plt.subplot(131)
                    plt.title('raw image')
                    plt.imshow(vimg[0][0].cpu().detach().numpy())

                    plt.subplot(132)
                    plt.title('gtcount:%.2f'%vgt_count)
                    plt.imshow(vden[0][0].cpu().detach().numpy())


                    plt.subplot(133)
                    plt.title('escount:%.2f'%ves_count)
                    plt.imshow(val_es_den[0][0].cpu().detach().numpy())
                    plt.savefig(saveimg+'/epoch{}-step{}.jpg'.format(epoch, index))


                    # plt.show()

            duration = time.time()-valstart
            valfps = index/duration
            valmae, valmse = eva_model(escount, gtcount)
            scheduler.step()
            epoch_duration = time.time() - epoch_start
            info = 'valloss:{%.8f} @ valmae:{%.3f} @ valmse:{%.3f} @ valfps{%.3f}\n ' % (
                valloss.avg ,
                valmae,
                valmse,
                valfps
            )
            logger.info(info)

            ###save model

            losssave = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict() if not args.multi_gpu else net.module.state_dict(),
                'best_loss': valloss.avg,
                'best_mae':args.best_mae,
                'best_mse':args.best_mse,
                'lr': get_learning_rate(optimizer)
            }

            if args.best_loss > valloss.avg:
                args.best_loss = valloss.avg
                torch.save(losssave, savemodel + '/best_loss_cut.tar')



            if args.best_mae > valmae:
                args.best_mae = valmae
                torch.save(losssave, savemodel + '/best_loss_mae.tar')


            if args.best_mse > valmse:
                args.best_mse = valmse
                torch.save(losssave, savemodel + '/best_loss_mse.tar')




        logger.info('right now train status:best_loss:%.2f-- best_mae:%.2f -- best_mse:%.2f ' % (
            best_loss * 10000, args.best_mae, args.best_mse))

        logger.info('epoch %d done,cost time %3f sec\n'%(epoch,epoch_duration))

        writer.add_scalars('data/loss', {
                'trainloss': trainloss.avg,
                'valloss': valloss.avg}, epoch)

        writer.add_scalars(args.method, {
                'valmse': valmse,
                'valmae': valmae,
                'trainmse': trainmse,
                'trainmae': trainmae
            }, epoch)


    logger.info(args.method + ' train complete')
    logger.info('best_loss:%.8f-- best_mae:%.2f -- best_mse:%.2f \n' % (best_loss , args.best_mae, args.best_mse))
    logger.info('save bestmodel to ' + savemodel)
    logger.info('#'*50)
    logger.info('\n')

