import mxnet as mx

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=True)
    act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
    return act

base_dim = 32
def get_symbol(num_classes, **kwargs):
    label = mx.symbol.Variable(name="label") # 224
    data = mx.symbol.Variable(name="data") # 224
    conv1 = Conv(data, num_filter=base_dim, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv1") # 224/112
    conv2_dw = Conv(conv1, num_group=base_dim, num_filter=base_dim, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv2_dw") # 112/112
    conv2 = Conv(conv2_dw, num_filter=base_dim*2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv2") # 112/112
    conv3_dw = Conv(conv2, num_group=base_dim*2, num_filter=base_dim*2, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv3_dw") # 112/56
    conv3 = Conv(conv3_dw, num_filter=base_dim*2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv3") # 56/56
    conv4_dw = Conv(conv3, num_group=base_dim*2, num_filter=base_dim*2, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv4_dw") # 56/56
    conv4 = Conv(conv4_dw, num_filter=base_dim*2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv4") # 56/56
    conv5_dw = Conv(conv4, num_group=base_dim*2, num_filter=base_dim*2, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv5_dw") # 56/28
    conv5 = Conv(conv5_dw, num_filter=base_dim*4, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv5") # 28/28
    conv6_dw = Conv(conv5, num_group=base_dim*4, num_filter=base_dim*4, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv6_dw") # 28/28
    conv6 = Conv(conv6_dw, num_filter=base_dim*4, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv6") # 28/28
    conv7_dw = Conv(conv6, num_group=base_dim*4, num_filter=base_dim*4, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv7_dw") # 28/14
    conv7 = Conv(conv7_dw, num_filter=base_dim*8, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv7") # 14/14

    conv8_dw = Conv(conv7, num_group=base_dim*8, num_filter=base_dim*8, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv8_dw") # 14/14
    conv8 = Conv(conv8_dw, num_filter=base_dim*8, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv8") # 14/14
    conv9_dw = Conv(conv8, num_group=base_dim*8, num_filter=base_dim*8, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv9_dw") # 14/14
    conv9 = Conv(conv9_dw, num_filter=base_dim*8, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv9") # 14/14
    conv10_dw = Conv(conv9, num_group=base_dim*8, num_filter=base_dim*8, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv10_dw") # 14/14
    conv10 = Conv(conv10_dw, num_filter=base_dim*8, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv10") # 14/14
    conv11_dw = Conv(conv10, num_group=base_dim*8, num_filter=base_dim*8, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv11_dw") # 14/14
    conv11 = Conv(conv11_dw, num_filter=base_dim*8, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv11") # 14/14
    conv12_dw = Conv(conv11, num_group=base_dim*8, num_filter=base_dim*8, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv12_dw") # 14/14
    conv12 = Conv(conv12_dw, num_filter=base_dim*8, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv12") # 14/14

    conv13_dw = Conv(conv12, num_group=base_dim*8, num_filter=base_dim*8, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv13_dw") # 14/7
    conv13 = Conv(conv13_dw, num_filter=base_dim*16, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv13") # 7/7
    conv14_dw = Conv(conv13, num_group=base_dim*16, num_filter=base_dim*16, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv14_dw") # 7/7
    conv14 = Conv(conv14_dw, num_filter=base_dim*16, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv14") # 7/7

    pool = mx.sym.Pooling(data=conv14, kernel=(7, 7), stride=(1, 1), pool_type="avg", name="global_pool", global_pool=True)
    fc = mx.symbol.FullyConnected(data=pool, num_hidden=num_classes, name='fc')
    cls_prob = mx.symbol.SoftmaxOutput(data=fc, label=label, multi_output=True, use_ignore=True,  name="cls_prob")
    group = mx.symbol.Group([cls_prob, label])
    return group
