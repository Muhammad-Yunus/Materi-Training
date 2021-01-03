import os
import datetime
import numpy as np

from keras.models import load_model
from keras.layers import Dense, Activation

from .dl_core_utils import Preprocessing, Evaluation, Ped
from .dl_core_optimizer import ModelOptimizer

import keras

class HistoryLog(keras.callbacks.Callback):
    def __init__(self):
        super(HistoryLog, self).__init__()
        self.socketio = None
        self.event = ''

    def set_socketio(self, socketio, event="feedback"):
        if self.socketio is None :
            self.socketio =  socketio
        if self.event == '' :
            self.event = event

    def on_epoch_end(self, epoch, logs=None):
        
        self.socketio.emit(self.event, 
                        "%s epoch %d : loss = %.2f, accuracy = %.2f%%" % 
                        (self.get_time(), epoch, logs["loss"], logs["accuracy"]*100)
                        )
        self.socketio.sleep(1)

    def get_time(self):
        return "[%s]" % datetime.datetime.now().strftime("%H:%M:%S.%f")

class TransferLearning(object):
    def __init__(self, 
                socketio, 
                event = "feedback",
                model_name="model-cnn-facerecognition.h5", 
                dim=13, 
                dataset="dataset/", 
                use_augmentation=True, 
                test_size=0.15, 
                val_size=0.15, 
                epoch=3, 
                batch=32):
        self.model = None
        self.socketio = socketio
        self.event = event
        self.model_name=model_name 
        self.dim=dim
        self.dataset=dataset 
        self.use_augmentation=use_augmentation 
        self.test_size=test_size 
        self.val_size=val_size
        self.epoch=epoch
        self.batch=batch
        self.is_running = False

        self.historyLog = HistoryLog()
        self.historyLog.set_socketio(self.socketio, event=self.event)

        self.modelOptimizer = ModelOptimizer()

    def init_model(self):
        self.model = load_model(self.model_name)

    def run(self):
        
        self.is_running = True
        self.socketio.emit(self.event, "%s <b>__Start Transfer Learning__<b>" % self.get_time())
        self.socketio.sleep(1)

        for i in range(len(self.model.layers)):
            if i > 6 :
                self.model.layers[i].trainable = True
            else :
                self.model.layers[i].trainable = False

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', 
            metrics=['accuracy'])

        self.model.pop()
        self.model.pop()

        self.model.add(Dense(self.dim))
        self.model.add(Activation("softmax"))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', 
            metrics=['accuracy'])

        self.socketio.emit(self.event, 
                        "%s finish modifying model %s" % 
                        (self.get_time(), self.model_name.split('\\')[-1]))
        self.socketio.sleep(1)

        self.model.summary(print_fn=lambda x: self.socketio.emit(self.event, x))

        # prepare dataset
        prepro = Preprocessing()
        names, images = prepro.load_dataset(dataset_folder = self.dataset)

        self.socketio.emit(self.event, 
                        "%s load dataset (%d sample / %d class) completed." % 
                        (self.get_time(), len(names), len(np.unique(names))))
        self.socketio.sleep(1)

        if self.use_augmentation :
            names, images = prepro.image_augmentator(images, names)
            self.socketio.emit(self.event, 
                            "%s augmenting dataset (%d sample / %d class) completed." % 
                            (self.get_time(), len(names), len(np.unique(names))))
            self.socketio.sleep(1)

        categorical_name_vec = prepro.convert_categorical(names)

        x_train, x_test, y_train, y_test = \
                        prepro.split_dataset(images, 
                                            categorical_name_vec, 
                                            test_size=self.test_size)

        self.socketio.emit(self.event, 
                        "%s split dataset (%d training set, %d test set) completed." % 
                        (self.get_time(), x_train.shape[0], x_test.shape[0]))
        self.socketio.sleep(1)

        #train model
        self.socketio.emit(self.event, "%s training model started." % self.get_time())
        self.socketio.sleep(1)

        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

        self.model.fit(x_train, 
                        y_train,
                        epochs=self.epoch,
                        batch_size=self.batch,
                        shuffle=True,
                        validation_split=self.val_size,
                        callbacks = [self.historyLog]
                        )

        self.socketio.emit(self.event, "%s training model completed." % self.get_time())
        self.socketio.sleep(1)

        self.socketio.emit(self.event, "%s starting model otimization for inference." % self.get_time())
        self.socketio.sleep(1)
        # optimize model for inference
        PATH = '\\'.join(self.model_name.split('\\')[:-1])

        self.modelOptimizer.h5_to_savedModel(model_name = self.model_name, 
                                            savedModel_folder = os.path.join(PATH, "tf_model"))
        
        self.socketio.emit(self.event, "%s saved model `tf_model/` created." % self.get_time())
        self.socketio.sleep(1)

        optimized_model_name = "frozen_graph_%s.pb" % self.get_datetime_str()
        self.modelOptimizer.optimize(savedModel_folder=os.path.join(PATH, "tf_model"), 
                                    target_name= os.path.join(PATH, optimized_model_name))
        
        self.socketio.emit(self.event, "%s optimized model %s created." % (self.get_time(), optimized_model_name))
        self.socketio.sleep(1)

        self.is_running = False

    def get_time(self):
        return "[%s]" % datetime.datetime.now().strftime("%H:%M:%S.%f")

    def get_datetime_str(self):
        return datetime.datetime.now().strftime("%d%m%Y_%H%M%S")