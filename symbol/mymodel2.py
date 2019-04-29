import mxnet as mx

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s' %(name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_bn' %(name, suffix), fix_gamma=False, eps=1e-3)
    #act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' %(name, suffix))
    act = mx.sym.LeakyReLU(data=bn, act_type="prelu", name='%s%s_relu' %(name, suffix))
    return act

def get_symbol(num_classes, **kwargs):
    label = mx.symbol.Variable(name="label") # 224
    data = mx.symbol.Variable(name="data") # 224
    conv1_ = Conv(data, num_filter=8, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv1_") # 224/112
    conv2_dw = Conv(conv1_, num_group=8, num_filter=8, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv2_dw") # 112/112
    conv2_sep = Conv(conv2_dw, num_filter=16, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv2_sep") # 112/112
    conv3_dw = Conv(conv2_sep, num_group=16, num_filter=16, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv3_dw") # 112/56
    conv3_sep = Conv(conv3_dw, num_filter=32, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv3_sep") # 56/56
    conv4_dw = Conv(conv3_sep, num_group=32, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv4_dw") # 56/56
    conv4_sep = Conv(conv4_dw, num_filter=16, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv4_sep") # 56/56
    conv5_dw = Conv(conv4_sep, num_group=16, num_filter=16, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv5_dw") # 56/28
    conv5_sep = Conv(conv5_dw, num_filter=32, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv5_sep") # 28/28
    conv6_dw = Conv(conv5_sep, num_group=32, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv6_dw") # 28/28
    conv6_sep = Conv(conv6_dw, num_filter=32, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv6_sep") # 28/28
    conv7_dw = Conv(conv6_sep, num_group=32, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv7_dw") # 28/14
    conv7_sep = Conv(conv7_dw, num_filter=64, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv7_sep") # 14/14

    conv8_dw = Conv(conv7_sep, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv8_dw") # 14/14
    conv8_sep = Conv(conv8_dw, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv8_sep") # 14/14
    conv9_dw = Conv(conv8_sep, num_group=128, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv9_dw") # 14/14
    conv9_sep = Conv(conv9_dw, num_filter=64, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv9_sep") # 14/14
    conv10_dw = Conv(conv9_sep, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv10_dw") # 14/14
    conv10_sep = Conv(conv10_dw, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv10_sep") # 14/14
    conv11_dw = Conv(conv10_sep, num_group=128, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv11_dw") # 14/14
    conv11_sep = Conv(conv11_dw, num_filter=64, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv11_sep") # 14/14
    conv12_dw = Conv(conv11_sep, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv12_dw") # 14/14
    conv12_sep = Conv(conv12_dw, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv12_sep") # 14/14

    conv13_dw = Conv(conv12_sep, num_group=128, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv13_dw") # 14/7
    conv13_sep = Conv(conv13_dw, num_filter=128, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv13_sep") # 7/7
    conv14_dw = Conv(conv13_sep, num_group=128, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(1, 1), name="conv14_dw") # 7/7
    conv14_sep = Conv(conv14_dw, num_filter=256, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv14_sep") # 7/7

    pool = mx.sym.Pooling(data=conv14_sep, kernel=(7, 7), stride=(1, 1), pool_type="avg", name="global_pool", global_pool=True)
    fc = mx.symbol.FullyConnected(data=pool, num_hidden=num_classes, name='fc')
    cls_prob = mx.symbol.SoftmaxOutput(data=fc, label=label, multi_output=True, use_ignore=True,  name="cls_prob")
    group = mx.symbol.Group([cls_prob, label])
    return group
