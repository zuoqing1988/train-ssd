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
    f1 = open('%s_anno_ori.txt'%lst_file, 'w')
    f2 = open('%s_prob.txt'%lst_file, 'w')
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
            f1.write('%s %d %d %d %d\n'%(img_name,x1,y1,x2,y2))
            f2.write('%s %f\n'%(img_name,1.0))
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
            f1.write('%s %d %d %d %d\n'%(img_name,x1,y1,x2,y2))
            f2.write('%s %f\n'%(img_name,1.0))

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare lists for dataset')
    parser.add_argument('--anno_file', dest='anno_file', help='dataset to use', type=str)
    parser.add_argument('--target', dest='target', help='target file',   type=str)
    parser.add_argument('--select', dest='select', help='select for val', type=bool, default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    anno2lst_one_class(args.anno_file,args.target,args.select)

