import mxnet as mx
import numpy as np


class Accuracy(mx.metric.EvalMetric):
    def __init__(self):
        super(Accuracy, self).__init__('acc')

    def update(self, labels, preds):
        # output: cls_prob_output
        # label: label
        pred_label = mx.ndarray.argmax_channel(preds[0]).asnumpy().astype('int32')
        label = labels[0].asnumpy()
        #print pred_label
        #print label
        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)
