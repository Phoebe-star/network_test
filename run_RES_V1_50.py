#python news.py --data_dir=data --batch_size=1 --mode=cmc
#python news.py --mode=test --image1=data/labeled/val/0046_00.jpg --image2=data/labeled/val/0049_07.jpg
import tensorflow as tf
import numpy as np
import cv2

#import cuhk03_dataset_label2
import big_dataset_label as cuhk03_dataset_label2


import random


from triplet_loss import batch_hard_triplet_loss



from importlib import import_module
from tensorflow.contrib import slim
from nets import NET_CHOICES
from heads import HEAD_CHOICES



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt  
from PIL import Image 

print tf.__version__
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', '80', 'batch size for training')
tf.flags.DEFINE_integer('max_steps', '210000', 'max steps for training')
tf.flags.DEFINE_string('logs_dir', 'logs_RES/', 'path to logs directory')
tf.flags.DEFINE_string('data_dir', 'data_eye/', 'path to dataset')
tf.flags.DEFINE_float('learning_rate', '0.01', '')
tf.flags.DEFINE_string('mode', 'train', 'Mode train, val, test')
tf.flags.DEFINE_string('image1', '', 'First image path to compare')
tf.flags.DEFINE_string('image2', '', 'Second image path to compare')

tf.flags.DEFINE_float('global_rate', '1.0', 'global rate')
tf.flags.DEFINE_float('local_rate', '1.0', 'local rate')
tf.flags.DEFINE_float('softmax_rate', '1.0', 'softmax rate')

tf.flags.DEFINE_integer('ID_num', '20', 'id number')
tf.flags.DEFINE_integer('IMG_PER_ID', '4', 'img per id')



tf.flags.DEFINE_integer('embedding_dim', '128', 'Dimensionality of the embedding space.')
tf.flags.DEFINE_string('initial_checkpoint', 'resnet_v1_50.ckpt', 'Path to the checkpoint file of the pretrained network.')





IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224



def preprocess(images, is_train):
    def train():    
        split = tf.split(images, [1, 1,1])
        shape = [1 for _ in xrange(split[0].get_shape()[1])]
        for i in xrange(len(split)):
            split[i] = tf.reshape(split[i], [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            split[i] = tf.image.resize_images(split[i], [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3])
            split[i] = tf.split(split[i], shape)
            for j in xrange(len(split[i])):
                #split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT , IMAGE_WIDTH , 3])
                split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3, 3])
                split[i][j] = tf.random_crop(split[i][j], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                split[i][j] = tf.image.random_flip_left_right(split[i][j])
                split[i][j] = tf.image.random_brightness(split[i][j], max_delta=32. / 255.)
                split[i][j] = tf.image.random_saturation(split[i][j], lower=0.5, upper=1.5)
                split[i][j] = tf.image.random_hue(split[i][j], max_delta=0.2)
                split[i][j] = tf.image.random_contrast(split[i][j], lower=0.5, upper=1.5)
                split[i][j] = tf.image.per_image_standardization(split[i][j])
        return [tf.reshape(tf.concat(split[0], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[2], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])]
    def val():
        split = tf.split(images, [1, 1,1])
        shape = [1 for _ in xrange(split[0].get_shape()[1])]
        for i in xrange(len(split)):
            split[i] = tf.reshape(split[i], [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            split[i] = tf.image.resize_images(split[i], [IMAGE_HEIGHT, IMAGE_WIDTH])
            split[i] = tf.split(split[i], shape)
            for j in xrange(len(split[i])):
                split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                split[i][j] = tf.image.per_image_standardization(split[i][j])
        return [tf.reshape(tf.concat(split[0], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])]
    return tf.cond(is_train, train, val)








def global_pooling(images1,weight_decay ):
    with tf.variable_scope('network_global_pool'):
        # Tied Convolution    
        global_pool = 7
    
        #conv1_branch1 = tf.layers.conv2d(images1, 512, [1, 1], reuse=None, name='conv1_branch1')        
        feat1_avg_pool1 = tf.nn.avg_pool(images1, ksize=[1, global_pool, global_pool, 1], strides=[1, 1, 1, 1], padding='VALID')
        #feat1_avg_pool1 = tf.nn.avg_pool(feat1_prod1, ksize=[1, global_pool, global_pool, 1], strides=[1, global_pool, global_pool, 1], padding='SAME')
        reshape_branch1 = tf.reshape(feat1_avg_pool1, [FLAGS.batch_size, -1])
        
        
    
        
        concat1_L2 = tf.nn.l2_normalize(reshape_branch1,dim=1)
        
     
        return concat1_L2                                                                                                                                                                                                        










def triplet_hard_loss(y_pred,id_num,img_per_id):
    with tf.variable_scope('hard_triplet'):

        SN = img_per_id  #img per id
        PN =id_num   #id num
        feat_num = SN*PN # images num
        
        y_pred = tf.nn.l2_normalize(y_pred,dim=1) 
    
        feat1 = tf.tile(tf.expand_dims(y_pred,0),[feat_num,1,1])
        feat2 = tf.tile(tf.expand_dims(y_pred,1),[1,feat_num,1])
        
        delta = tf.subtract(feat1,feat2)
        dis_mat = tf.reduce_sum(tf.square(delta), 2)+ 1e-8

        dis_mat = tf.sqrt(dis_mat)
     
        #dis_mat = tf.reduce_sum(tf.square(tf.subtract(feat1, feat2)), 2)
        #dis_mat = tf.sqrt(dis_mat)
        

    
        positive = dis_mat[0:SN,0:SN]
        negetive = dis_mat[0:SN,SN:]
        
        for i in range(1,PN):
            positive = tf.concat([positive,dis_mat[i*SN:(i+1)*SN,i*SN:(i+1)*SN]],axis = 0)
            if i != PN-1:
                negs = tf.concat([dis_mat[i*SN:(i+1)*SN,0:i*SN],dis_mat[i*SN:(i+1)*SN, (i+1)*SN:]],axis = 1)
            else:
                negs = tf.concat(dis_mat[i*SN:(i+1)*SN, 0:i*SN],axis = 0)
            negetive = tf.concat([negetive,negs],axis = 0)
  
        p=positive
        n=negetive
        positive = tf.reduce_max(positive,1)
        negetive = tf.reduce_min(negetive,axis=1) #acc
        
        #negetive = tf.reduce_mean(negetive,1)
        #negetive = tf.reduce_max(negetive,axis=1) #false

        a1 = 0.3
        
        basic_loss = tf.add(tf.subtract(positive,negetive), a1)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        #loss = tf.reduce_mean(tf.maximum(0.0,positive-negetive+a1))
       
        return loss ,tf.reduce_mean(positive) ,tf.reduce_mean(negetive)
        

        
        
        
        

        


def main(argv=None):

    
    
    
    if FLAGS.mode == 'test':
        FLAGS.batch_size = 1
    
    if FLAGS.mode == 'cmc':
        FLAGS.batch_size = 1

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
  
    images = tf.placeholder(tf.float32, [3, FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images')
    
    images_total = tf.placeholder(tf.float32, [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images_total')
    
    labels = tf.placeholder(tf.float32, [FLAGS.batch_size], name='labels')
 
    

    
    
    
    is_train = tf.placeholder(tf.bool, name='is_train')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    weight_decay = 0.0005
    tarin_num_id = 0
    val_num_id = 0

    if FLAGS.mode == 'train':
        tarin_num_id = cuhk03_dataset_label2.get_num_id(FLAGS.data_dir, 'train')
        print(tarin_num_id, '               11111111111111111111               1111111111111111')
    elif FLAGS.mode == 'val':
        val_num_id = cuhk03_dataset_label2.get_num_id(FLAGS.data_dir, 'val')
    #images1, images2,images3 = preprocess(images, is_train)
    
    

    
    
    
    
    
    
    # Create the model and an embedding head.
    model = import_module('nets.' + 'resnet_v1_50')
    head = import_module('heads.' + 'fc1024')
    
    
    # Feed the image through the model. The returned `body_prefix` will be used
    # further down to load the pre-trained weights for all variables with this
    # prefix.
    endpoints, body_prefix = model.endpoints(images_total, is_training=True)

    with tf.name_scope('head'):
        endpoints = head.head(endpoints, FLAGS.embedding_dim, is_training=True)
    
    
    
    print 'model_output : ',endpoints['model_output'] # (bt,2048)
    print 'global_pool : ',endpoints['global_pool'] # (bt,2048)
    print 'resnet_v1_50/block4 : ',endpoints['resnet_v1_50/block4']# (bt,7,7,2048)
    #  see   net.resnet_V1   line 258 
    print ' 1\n'
    

    
    
    
    
    train_mode = tf.placeholder(tf.bool)
  


    print('Build network')
    
    feat = endpoints['resnet_v1_50/block4']# (bt,7,7,2048)

    #feat = tf.convert_to_tensor(feat, dtype=tf.float32)
    # global
    feature = global_pooling(feat,weight_decay)
    #loss_triplet,PP,NN = triplet_hard_loss(feature,FLAGS.ID_num,FLAGS.IMG_PER_ID)
    loss_triplet ,PP,NN = batch_hard_triplet_loss(labels,feature,0.3)

    
    

   
    
    
    
    
    
    loss =  loss_triplet*FLAGS.global_rate
  
 
    
    
    
    
    
    
    
    
    # These are collected here before we add the optimizer, because depending
    # on the optimizer, it might add extra slots, which are also global
    # variables, with the exact same prefix.
    model_variables = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, body_prefix)
    
    
    
    
    #optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    #optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    #train = optimizer.minimize(loss, global_step=global_step)
    
    
    
    
    # Update_ops are used to update batchnorm stats.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #train_op = optimizer.minimize(loss_mean, global_step=global_step)

    
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        train = optimizer.minimize(loss, global_step=global_step)
    

    lr = FLAGS.learning_rate

    #config=tf.ConfigProto(log_device_placement=True)
    #config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)) 
    # GPU
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    
    with tf.Session(config=config) as sess:
        
        

        
        
        print '\n'
        #print model_variables
        print '\n'
        #sess.run(tf.global_variables_initializer())
        #saver = tf.train.Saver()
        
        #checkpoint_saver = tf.train.Saver(max_to_keep=0)
        checkpoint_saver = tf.train.Saver()


        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore model')
            print ckpt.model_checkpoint_path
            #saver.restore(sess, ckpt.model_checkpoint_path)
            checkpoint_saver.restore(sess, ckpt.model_checkpoint_path)
                    
        #for first , training load imagenet
        else:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(model_variables)
            print FLAGS.initial_checkpoint
            saver.restore(sess, FLAGS.initial_checkpoint)
            
         
            
         
   
            
            
        if FLAGS.mode == 'train':
            step = sess.run(global_step)
            for i in xrange(step, FLAGS.max_steps + 1):

                batch_images, batch_labels, batch_images_total = cuhk03_dataset_label2.read_data(FLAGS.data_dir, 'train', tarin_num_id,
                    IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS.batch_size,FLAGS.ID_num,FLAGS.IMG_PER_ID)
              
                feed_dict = {learning_rate: lr,  is_train: True , train_mode: True, images_total: batch_images_total, labels: batch_labels}
             
                
                
                
                _,train_loss = sess.run([train,loss], feed_dict=feed_dict) 
                    
                print('Step: %d, Learning rate: %f, Train loss: %f ' % (i, lr, train_loss))
                
                gtoloss,gp,gn = sess.run([loss_triplet,PP,NN], feed_dict=feed_dict)   
                print 'global hard: ',gtoloss
                print 'global P: ',gp
                print 'global N: ',gn
                
                

       
                
                
                #lr = FLAGS.learning_rate / ((2) ** (i/160000)) * 0.1
                lr = FLAGS.learning_rate * ((0.0001 * i + 1) ** -0.75)
                if i % 100 == 0:
                    #saver.save(sess, FLAGS.logs_dir + 'model.ckpt', i)

                    checkpoint_saver.save(sess,FLAGS.logs_dir + 'model.ckpt', i)
                
                
                
                
        

        elif FLAGS.mode == 'val':
            total = 0.
            for _ in xrange(10):
                batch_images, batch_labels = cuhk03_dataset_label2.read_data(FLAGS.data_dir, 'val', val_num_id,
                    IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS.batch_size)
                feed_dict = {images: batch_images, labels: batch_labels, is_train: False}
                prediction = sess.run(inference, feed_dict=feed_dict)
                prediction = np.argmax(prediction, axis=1)
                label = np.argmax(batch_labels, axis=1)

                for i in xrange(len(prediction)):
                    if prediction[i] == label[i]:
                        total += 1
            print('Accuracy: %f' % (total / (FLAGS.batch_size * 10)))

            '''
            for i in xrange(len(prediction)):
                print('Prediction: %s, Label: %s' % (prediction[i] == 0, labels[i] == 0))
                image1 = cv2.cvtColor(batch_images[0][i], cv2.COLOR_RGB2BGR)
                image2 = cv2.cvtColor(batch_images[1][i], cv2.COLOR_RGB2BGR)
                image = np.concatenate((image1, image2), axis=1)
                cv2.imshow('image', image)
                key = cv2.waitKey(0)
                if key == 1048603:  # ESC key
                    break
            '''

        
        elif FLAGS.mode == 'cmc':    
          do_times = 1
          cmc_sum=np.zeros((100, 100), dtype='f')
          for times in xrange(do_times):  
              path = 'data' 
              set = 'train'
              
              cmc_array=np.ones((100, 100), dtype='f')
              
              batch_images = []
              batch_labels = []
              index_gallery_array=np.ones((1, 100), dtype='f')
              gallery_bool = True
              probe_bool = True
              for j in xrange(100):
                      id_probe = j
                      for i in xrange(100):
                              batch_images = []
                              batch_labels = []
                              filepath = ''
                              
                              #filepath_gallery = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, i, index_gallery)
                              #filepath_probe = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, id_probe, index_probe)                          
                              
                              if gallery_bool == True:
                                    while True:
                                          index_gallery = int(random.random() * 10)
                                          index_gallery_array[0,i] = index_gallery
  
                                          filepath_gallery = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, i, index_gallery)
                                          if not os.path.exists(filepath_gallery):
                                              continue
                                          break
                              if i ==99:
                                  gallery_bool = False
                              if gallery_bool == False:
                                          index_gallery = index_gallery_array[0,i]
                                          filepath_gallery = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, i, index_gallery)
                              
                              
                              
                              if probe_bool == True:
                                    while True:
                                          index_probe = int(random.random() * 10)
                                          filepath_probe = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, id_probe, index_probe)
                                          if not os.path.exists(filepath_probe):
                                              continue
                                          if index_gallery_array[0,id_probe] == index_probe:
                                              continue
                                          probe_bool = False
                                          break
                              if i ==99:
                                  probe_bool = True
                              
                              
                              '''
                              while True:
                                    index_probe = int(random.random() * 10)
                                    filepath_probe = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, id_probe, index_probe)
                                    if not os.path.exists(filepath_gallery):
                                        continue
                                    if index_gallery_array[1,id_probe] == index_probe:
                                        continue
                                    break
                              '''
                              
                              #filepath_gallery = 'data/labeled/val/0000_01.jpg'
                              #filepath_probe   = 'data/labeled/val/0000_02.jpg'
                                                                          
                              image1 = cv2.imread(filepath_gallery)
                              image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
                              image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                              image1 = np.reshape(image1, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
                              
                              image2 = cv2.imread(filepath_probe)
                              image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
                              image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                              image2 = np.reshape(image2, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
                              
                              test_images = np.array([image1, image2, image2])
                              
                              #print (filepath_gallery)
                              #print (filepath_probe)
                              #print ('1111111111111111111111')
          
                              if i == j:
                                  batch_labels = [1., 0.]
                              if i != j:    
                                  batch_labels = [0., 1.]
                              batch_labels = np.array(batch_labels)
                              print('test  img :',test_images.shape)
                              
                              feed_dict = {images: test_images, is_train: False}
                              prediction = sess.run(DD, feed_dict=feed_dict)
                              #print (prediction, prediction[0][1])
                              
                              print (filepath_gallery,filepath_probe)
                              
                              #print(bool(not np.argmax(prediction[0])))
                              print (prediction)
                              
                              cmc_array[j,i] = prediction
                              
                              #print(i,j)
                             
                              
                              #prediction = sess.run(inference, feed_dict=feed_dict)
                              #prediction = np.argmax(prediction, axis=1)
                              #label = np.argmax(batch_labels, axis=1)
                              
  
              
              cmc_score = cmc.cmc(cmc_array)
              cmc_sum = cmc_score + cmc_sum
              print(cmc_score)
          cmc_sum = cmc_sum/do_times
          print(cmc_sum)
          print('final cmc') 
        
        
        
        elif FLAGS.mode == 'test':
            image1 = cv2.imread(FLAGS.image1)
            image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image1 = np.reshape(image1, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
            image2 = cv2.imread(FLAGS.image2)
            image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            image2 = np.reshape(image2, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
            test_images = np.array([image1, image2,image2])

            feed_dict = {images: test_images, is_train: False, droup_is_training: False}
            #prediction, prediction2 = sess.run([DD,DD2], feed_dict=feed_dict)
            prediction = sess.run([inference], feed_dict=feed_dict)
            prediction = np.array(prediction)
            print prediction.shape
            print( np.argmax(prediction[0])+1)
            
           
        
            #print(bool(not np.argmax(prediction[0])))

if __name__ == '__main__':
    tf.app.run()