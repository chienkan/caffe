#!/usr/bin/env python

from caffe import Net
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf as pb
import google.protobuf.text_format

# 1.generate prototxt file
net = caffe_pb2.NetParameter()
with open('newmodel.prototxt', 'r') as f:
    pb.text_format.Merge(f.read(), net)

suffix = '_report'

for layer in net.layer:
    layer.name += suffix
    for i in range(len(layer.bottom)):
        layer.bottom[i] += suffix
    for i in range(len(layer.top)):
        layer.top[i] += suffix

with open('newmodel_report.prototxt', 'w') as f:
    f.write(pb.text_format.MessageToString(net))

# 2.generate caffemodel file
net_old = Net('newmodel.prototxt', 'newmodel.caffemodel', 0)
net_new = Net('newmodel_report.prototxt', "newmodel.caffemodel", 0)

for layer_name in net_old.params.keys():
#    print layer_name
    layer_new_name = layer_name + suffix
    net_new.params[layer_new_name] = net_old.params[layer_name]
net_new.save('newmodel_report.caffemodel')
