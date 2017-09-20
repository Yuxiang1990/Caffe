import caffe
import sys
sys.path.append("/home/dn/caffe_project/")
import numpy as np
from numpy import round,clip,equal,sum

def sensitivity(pred, ptrue):
    total_num = 0
    target_count = 0
    pred = np.squeeze(pred)
    ptrue = np.squeeze(ptrue)
    p_target = clip(pred * ptrue,0,1) #batch_size 1 48 48 48

    for i in range(ptrue.shape[0]):
        if(sum(ptrue[i]) > 1):
            total_num +=1
        if(sum(p_target[i] > 0.1)):
            target_count +=1
    
    return target_count /( total_num * 1.0)

def metric_yyx(pred, ptrue,model_type):
    
    beta = 1 # when beta <1 ,recall in fmeasure is more important
    pred = np.squeeze(pred).flatten()
    ptrue = np.squeeze(ptrue).flatten()

    accuracy = equal(round(pred),ptrue).mean()

    x1 = sum(round(clip(pred * ptrue,0,1)))
    x2 = sum(round(clip(ptrue,0,1)))
    x3 = sum(round(clip(pred,0,1)))

    recall = x1 / x2
    precision = x1 / x3
    bb = beta**2
    fmeasure = (1.0 + bb) * (recall * precision) / (bb * precision + recall)
    if model_type == "resnet":
        return accuracy, recall, precision, fmeasure
    else:
        senti= sensitivity(pred, ptrue)
        return accuracy, recall, precision, fmeasure,senti

"""
    if key == 'acc':
        return equal(round(pred),ptrue).mean()
    elif key == 'recall':
        return sum(round(clip(pred * ptrue,0,1))) / sum(round(clip(ptrue,0,1)))
    elif key == 'precision':
        return sum(round(clip(pred * ptrue,0,1))) / sum(round(clip(pred,0,1)))
    elif key == 'fmeasure':
        recall = sum(round(clip(pred * ptrue,0,1))) / sum(round(clip(ptrue,0,1)))
        precision = sum(round(clip(pred * ptrue,0,1))) / sum(round(clip(pred,0,1)))
        bb = beta**2
        return (1.0 + bb) * (recall * precision) / (bb * precision + recall)
"""
def metric_yyx1(pred, ptrue,key):
    beta = 1 # when beta <1 ,recall in fmeasure is more important
    pred = np.squeeze(pred).flatten()
    ptrue = np.squeeze(ptrue).flatten()
    if key == 'acc':
        return equal(round(pred),ptrue).mean()
    elif key == 'recall':
        return sum(round(clip(pred * ptrue,0,1))) / sum(round(clip(ptrue,0,1)))
    elif key == 'precision':
        return sum(round(clip(pred * ptrue,0,1))) / sum(round(clip(pred,0,1)))
    elif key == 'fmeasure':
        recall = sum(round(clip(pred * ptrue,0,1))) / sum(round(clip(ptrue,0,1)))
        precision = sum(round(clip(pred * ptrue,0,1))) / sum(round(clip(pred,0,1)))
        bb = beta**2
        return (1.0 + bb) * (recall * precision) / (bb * precision + recall)

class Metrics_new(caffe.Layer):#bottom[0]:score bottom[1]:label
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.model_type = params['model_type']
        if len(bottom) !=2:
            raise Exception("yyx:Need two inputs to compute the acc!")

    def reshape(self, bottom, top):
        if bottom[0].data.shape != bottom[1].data.shape:
            print(bottom[0].data.shape)
            print(bottom[1].data.shape)
            raise Exception("yyx:the dimension of inputs should match")
        top[0].reshape(1)
        top[1].reshape(1)
        top[2].reshape(1)
        top[3].reshape(1)
        if not self.model_type  == "resnet":
            top[4].reshape(1)

    def forward(self, bottom, top):
        #top[0].data[0] = metric_yyx(bottom[0].data, bottom[1].data,self.key)
        if  self.model_type  == "resnet":
            top[0].data[0], top[1].data[0], top[2].data[0], top[3].data[0] = metric_yyx(bottom[0].data, bottom[1].data,self.model_type)
        else:
            top[0].data[0], top[1].data[0], top[2].data[0], top[3].data[0],top[4].data[0] = metric_yyx(bottom[0].data, bottom[1].data,self.model_type)
        pass

    def backward(self, top, propagate_down, bottom):
        pass

if __name__ == '__main__':
    label = np.load("/home/dn/caffe_project/bug/label.npy")
    result = np.load("/home/dn/caffe_project/bug/result.npy")

    accuracy, recall, precision, fmeasure, senti = metric_yyx(result, label)
    print(accuracy, recall, precision, fmeasure, senti)
    pass
