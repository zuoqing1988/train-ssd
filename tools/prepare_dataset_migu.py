from __future__ import print_function
import sys, os
import argparse
import subprocess
import numpy as np
import numpy.random as npr
import mxnet
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, '..'))

def anno2lst_one_class(anno_file, lst_file, select=False):
    with open(anno_file, 'r') as f:
        anno_lines = f.readlines()
    num_lines = len(anno_lines)
    with open(lst_file, 'w') as f:
        if select:
            select_num = min(5000,num_lines)
            keep = npr.choice(num_lines, size=select_num, replace=False)
            idx = 0
            for i in keep:
                cur_line = anno_lines[i].split()
                img_name = cur_line[0]
                class_name = cur_line[1]
                width = float(cur_line[2])
                height = float(cur_line[3])
                x1 = float(cur_line[4])
                y1 = float(cur_line[5])
                x2 = float(cur_line[6])
                y2 = float(cur_line[7])
                px = float(cur_line[8])
                py = float(cur_line[9])
                f.write('%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%s\n'%(idx,2,6,0,x1/width,y1/height,x2/width,y2/height,0,img_name))
                idx = idx + 1
        else:
            for i in range(num_lines):
                cur_line = anno_lines[i].split()
                img_name = cur_line[0]
                class_name = cur_line[1]
                width = float(cur_line[2])
                height = float(cur_line[3])
                x1 = float(cur_line[4])
                y1 = float(cur_line[5])
                x2 = float(cur_line[6])
                y2 = float(cur_line[7])
                px = float(cur_line[8])
                py = float(cur_line[9])
                f.write('%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%s\n'%(i,2,6,0,x1/width,y1/height,x2/width,y2/height,0,img_name))

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare lists for dataset')
    parser.add_argument('--anno_file', dest='anno_file', help='dataset to use', type=str)
    parser.add_argument('--target', dest='target', help='output list file',   type=str)
    parser.add_argument('--root', dest='root_path', help='dataset root path',   type=str)
    parser.add_argument('--select', dest='select', help='select for val', type=bool, default=False)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle list', type=bool, default=True)
    parser.add_argument('--true-negative', dest='true_negative', help='use images with no GT as true_negative', type=bool, default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    anno2lst_one_class(args.anno_file,args.target,args.select)

    im2rec_path = os.path.join(mxnet.__path__[0], 'tools/im2rec.py')
    # final validation - sometimes __path__ (or __file__) gives 'mxnet/python/mxnet' instead of 'mxnet'
    if not os.path.exists(im2rec_path):
        im2rec_path = os.path.join(os.path.dirname(os.path.dirname(mxnet.__path__[0])), 'tools/im2rec.py')
    if args.shuffle:
        subprocess.check_call(["python", im2rec_path,
            os.path.abspath(args.target), os.path.abspath(args.root_path), "--pack-label"])
    else:
        subprocess.check_call(["python", im2rec_path,
            os.path.abspath(args.target), os.path.abspath(args.root_path), "--no-shuffle --pack-label"])
    print("Record file {} generated...".format(args.target.split('.')[0] + '.rec'))
