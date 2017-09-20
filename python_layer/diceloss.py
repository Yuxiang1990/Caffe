import caffe
import numpy as np
from numpy import round,clip,equal,sum

""" pt 
layer {
  type: 'Python'
  name: 'loss'
  top: 'loss'
  bottom: 'ipx'
  bottom: 'ipy'
  python_param {
    module: 'loss' # the module name -- usually the filename -- that needs to be in $PYTHONPATH
    layer: 'DiceLoss_layer' # the layer name -- the class name in the module
  }
  loss_weight: 1 # set loss weight so Caffe knows this is a loss layer
}
"""

def dice_coef_loss(y_true, y_pred):
    sum=y_true.sum()+y_pred.sum()+1.
    dice=(2.* (y_true * y_pred).sum()+1.)/sum
    return 1.- dice
    # smooth = 1
    # y_true_f = np.squeeze(y_true).flatten()
    # y_pred_f = np.squeeze(y_pred).flatten()
    
    # intersection = sum(y_true_f * y_pred_f)
    # return -(2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)

class DiceLoss_layer(caffe.Layer):
    """
    Compute energy based on dice coefficient.
    """
    union = None
    intersection = None
    result = None
    gt = None

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        self.batch_size=bottom[1].data.shape[0]


    def reshape(self, bottom, top):
        # check input dimensions match
        # print bottom[0].data.shape
        # print bottom[1].data.shape
        if bottom[0].count!=bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        self.diff=np.zeros_like(bottom[0].data,dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)


    def forward(self, bottom, top):
        # top[0].data[0] = dice_coef_loss(bottom[0].data[:,0], bottom[1].data)

        self.diff[...]=bottom[1].data
        # self.sum=np.ndarray(self.batch_size)
        # self.dice=np.ndarray(self.batch_size)
        # for idx in range(self.batch_size):
        #     self.sum[idx]=bottom[0].data[idx].sum()+bottom[1].data[idx].sum()+1.
        #     self.
            
        self.sum=bottom[0].data.sum()+bottom[1].data.sum()+1.
        self.dice=(2.* (bottom[0].data * bottom [1].data).sum()+1.)/self.sum
        top[0].data[...] = 1.- self.dice


    def backward(self, top, propagate_down, bottom):
        if propagate_down[1]:
            raise Exception("label not diff")
        elif propagate_down[0]:
            bottom[0].diff[...] = (-2.*self.diff+self.dice)/self.sum
        else:
            raise Exception("no diff")

if __name__ == '__main__':
    data_1 = np.random.random([3,3,3])
    data_2 = data_1 > 0.5
    dice_loss = dice_coef_loss(data_1, data_2)
    print dice_loss
    pass