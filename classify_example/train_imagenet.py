import argparse
import mxnet as mx
import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+'/symbol')
from classify_core.imdb import IMDB
from train import train_net
import mymodel

def train_imagenet(anno_file, color_mode, num_classes, prefix, ctx,
                pretrained, epoch, begin_epoch, end_epoch, batch_size, thread_num, 
                frequent, lr,lr_epoch, resume):
    imdb = IMDB(anno_file)
    gt_imdb = imdb.get_annotations()
    sym = mymodel.get_symbol(num_classes)

    train_net(sym, prefix, ctx, pretrained, epoch, begin_epoch, end_epoch, gt_imdb, color_mode, batch_size, thread_num,
              frequent, not resume, lr, lr_epoch)

def parse_args():
    parser = argparse.ArgumentParser(description='Train O_net(48-net)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--anno_file', dest='anno_file', help='anno file',
                        default='classify_data/anno.txt', type=str)
    parser.add_argument('--color_mode', dest='color_mode', help='color_mode bgr or rgb',
                        default='rgb', type=str)
    parser.add_argument('--num_classes', dest='num_classes', help='num of classes',
                        default=1000, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default='model/mymodel', type=str)
    parser.add_argument('--gpus', dest='gpu_ids', help='GPU device to train with',
                        default='0', type=str)
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained prefix',
                        default='model/mymodel', type=str)
    parser.add_argument('--epoch', dest='epoch', help='load epoch',
                        default=0, type=int)
    parser.add_argument('--begin_epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=16, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='batch_size of training',
                        default=512, type=int)
    parser.add_argument('--thread_num', dest='thread_num', help='thread num of training',
                        default=4, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=100, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate',
                        default=0.01, type=float)
    parser.add_argument('--lr_epoch', dest='lr_epoch', help='learning rate epoch',
                        default='8,14', type=str)
    parser.add_argument('--resume', dest='resume', help='continue training', action='store_true')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    ctx = [mx.gpu(int(i)) for i in args.gpu_ids.split(',')]
    lr_epoch = [int(i) for i in args.lr_epoch.split(',')]
    train_imagenet(args.anno_file, args.color_mode, args.num_classes, args.prefix, ctx, args.pretrained, args.epoch, args.begin_epoch, 
                args.end_epoch, args.batch_size, args.thread_num, args.frequent, args.lr, lr_epoch, args.resume)
