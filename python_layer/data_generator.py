import caffe
import sys
sys.path.append("/home/dn/caffe_project/")
import numpy as np
from processing_api.api import prepare_for_unet3D

def image_generator(batch_size, path, data_prefix,model_type, shuffle = True):
    which_sample_select = 0
    batch_index = 0
    unet_input = []
    unet_output = []
    while True:
        print('Loading and preprocessing ' + data_prefix + ' data...',which_sample_select)
        imgs_train = np.load(path + data_prefix + "_" + str(which_sample_select).rjust(4,'0')+".npz")['arr_0'].astype(np.float32)
        imgs_neg_train = np.load(path + data_prefix+ "_neg_"+str(which_sample_select).rjust(4,'0')+".npz")['arr_0'].astype(np.float32)
        if model_type == 'resnet':
            imgs_pos_class = np.ones(len(imgs_train))
            imgs_neg_class = np.zeros(len(imgs_neg_train))
        elif model_type == 'unet':
            imgs_pos_class = np.load(path + data_prefix + "_mask_" + str(which_sample_select).rjust(4,'0')+".npz")['arr_0'].astype(np.float32)
            imgs_neg_class = np.zeros(imgs_neg_train.shape)
        
        which_sample_select +=1
        if(which_sample_select == 30):
            which_sample_select = 0
            
        print(imgs_train.shape,imgs_pos_class.shape)
        print(imgs_neg_train.shape,imgs_neg_class.shape)
    
        imgs_train = prepare_for_unet3D(imgs_train)
        imgs_neg_train = prepare_for_unet3D(imgs_neg_train)
        
        pos_data = imgs_train
        pos_data_mask = imgs_pos_class
        neg_data = imgs_neg_train
        neg_data_mask = imgs_neg_class
        pos_len = len(pos_data)# input len = pos_len *2
        neg_len = len(neg_data)
        
        if shuffle:
            rand_pos = np.random.choice(range(pos_len), pos_len, replace=False)
            rand_neg = np.random.choice(range(neg_len), neg_len, replace=False)
            pos_data = pos_data[rand_pos]
            pos_data_mask = pos_data_mask[rand_pos]
            
            neg_data = neg_data[rand_neg]
            neg_data_mask = neg_data_mask[rand_neg]
            
        for i in range(min(pos_len, neg_len) * 2):
            if(i%2):
                unet_input.append(pos_data[i//2])
                unet_output.append(pos_data_mask[i//2])
                batch_index +=1
            else:
                unet_input.append(neg_data[i//2])
                unet_output.append(neg_data_mask[i//2])
                batch_index +=1
            if batch_index >=batch_size:
                x = np.array(unet_input)
                y = np.array(unet_output)
                # print("generate batchsize data")
                while(1):
                    yield x,y
                unet_input = []
                unet_output = []
                batch_index = 0

class DataGeneratorlayer(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['data', 'label']
        params = eval(self.param_str)
        self.batch_size = params['batchsize']
        self.path = params['data_generate_path']
        self.cubesize = params['cubesize']
        self.data_prefix = params['data_prefix']
        self.model_type = params['model_type']

        self.img_generate = image_generator(self.batch_size, self.path, self.data_prefix,
                                                                self.model_type, shuffle = True)
        print("DataGeneratorlayer", params)
        print('python layer setup ok ')

    def forward(self, bottom, top):
        data_generate = self.img_generate.next()
        top[0].data[...] = np.expand_dims(data_generate[0][:,24,:,:,0],axis=1)
        if(self.model_type == "resnet"):
            top[1].data[...] = np.expand_dims(data_generate[1],axis=-1)
        elif(self.model_type == "unet"):
            top[1].data[...] = np.expand_dims(data_generate[1][:,24,:,:,0],axis=1)
        print('python layer forward ok')
        sys.stdout.flush()
    
    
    def reshape(self, bottom, top):
        # print('reshape')
        top[0].reshape(*(self.batch_size,1,self.cubesize,self.cubesize))#,self.cubesize))# *????
        if(self.model_type == "resnet"):
            top[1].reshape(self.batch_size,1)
        elif(self.model_type == "unet"):
            top[1].reshape(self.batch_size,1,self.cubesize,self.cubesize)#,self.cubesize)
        # print('python layer reshape ok')
    
    
    def backward(self, top, propagate_down, bottom):
        pass

if __name__ == '__main__':
    example_path = '/home/dn/tianchi/final_unet_npy_resnet_600_36/'
    data = image_generator(1000,example_path,shuffle = True)
    for _ in range(10):    
        print(data.next()[0][:,:,:,16,0].shape,data.next()[1].shape)
    print('data generator ok')