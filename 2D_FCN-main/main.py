import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime
import json
import os
import cv2

from models.FCN import make_model
from utils.focal import focal_loss
from utils.dataloader import generate_dataset
from tensorflow.keras.layers import *
from tensorflow.keras import backend
from utils.datadisplay import show_data

if __name__ == '__main__':
    ROOT = 'Dataset/'
    WORK_ON = 'off_road'  # 'on_road' #'off_road' #
    NETWORK = '3D-FCN2'  # '3D-FCN'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    X_train, y_train, X_test, y_test = generate_dataset(WORK_ON)
    image_shape = X_train.shape[2:]
    n_classes = 2
    input_shape = X_train.shape[1:]
    img_input = Input(shape=(input_shape), name='input')
    kernel_size = (3, 3, 3)
    pool_size = (2, 2, 2)

    filter_size = 64

    opt = 'SGD'  # 'Adam' doesn't work
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']

    no_tests = 1
    epoch_num = 40  # 80 #18 #240
    loss_name = 'focal'  # 'loss'
    for i in range(no_tests):
        print('################################################')
        print('Test number %d' % i)
        backend.clear_session()
        model = make_model(input_shape, n_classes, filter_size, kernel_size, pool_size)  # reset model each time
        trained_weights = outfile = os.path.join('2DFCN2ECCV.h5')
        if os.path.exists(outfile):
            model.load_weights(trained_weights)
            print('Load trained weights')
        if loss_name.lower() == 'loss':
            model.compile(optimizer=opt, loss=loss, metrics=metrics)
        elif loss_name.lower() == 'focal':
            model.compile(optimizer=opt, loss=focal_loss(alpha=.25, gamma=2), metrics=metrics)
        #        model.compile(optimizer=opt, loss=focal_loss, metrics=metrics)

        history = model.fit(X_train, y_train, epochs=epoch_num, validation_data=(X_test, y_test), verbose=1,
                            batch_size=1)

        # In[24]:

        currentDT = datetime.datetime.now()
        dt_str = currentDT.strftime("%Y-%m-%d_%H-%M-%S")
        # model_fname = '%s_%s_%s_%s.h5' % (NETWORK, WORK_ON, loss_name, dt_str)
        model_fname = '2DFCN2ECCV.h5'
        # save model
        model.save(model_fname)
        print('Saved model to ' + model_fname)
        # load model
        # model = load_model(model_fname)

        # In[17]:

        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        xc = range(epoch_num)

        out_dic = {'train_loss': [str(num) for num in train_loss],
                   'val_loss': [str(num) for num in val_loss],
                   'train_acc': [str(num) for num in train_acc],
                   'val_acc': [str(num) for num in val_acc],
                   'epoch': [int(num) for num in xc]
                   }
        history_fname = '%s_%s_%s_%s_history.json' % (NETWORK, WORK_ON, loss_name, dt_str)
        with open(history_fname, 'w') as fp:
            fp.write(json.dumps(out_dic, indent=4, sort_keys=True))

        fig = plt.figure(1, figsize=(7, 5))
        fig.clf()
        plt.plot(xc, train_loss)
        plt.plot(xc, val_loss)
        plt.xlabel('Num of Epochs')
        plt.ylabel('Loss')
        plt.title('train-loss vs val-loss')
        plt.grid(True)
        plt.legend(['Train', 'Val'])
        plt.style.use(['classic'])
        #    plt.xlim([0, epoch_num])
        #    plt.ylim([0, 1])
        fig1_fname = '%s_%s_%s_%s_fig1.png' % (NETWORK, WORK_ON, loss_name, dt_str)
        plt.savefig(fig1_fname, dpi=300)

        fig2 = plt.figure(2, figsize=(7, 5))
        fig2.clf()
        plt.plot(xc, train_acc)
        plt.plot(xc, val_acc)
        plt.xlabel('Num of Epochs')
        plt.ylabel('Accuracy')
        plt.title('train-acc vs val-acc')
        plt.grid(True)
        plt.legend(['Train', 'Val'], loc=4)
        # print plt.style.available # use bmh, classic,ggplot for big pictures
        plt.style.use(['classic'])
        #    plt.xlim([0, epoch_num])
        #    plt.ylim([0, 1])
        fig2_fname = '%s_%s_%s_%s_fig2.png' % (NETWORK, WORK_ON, loss_name, dt_str)
        plt.savefig(fig2_fname, dpi=300)

        # # Evaluation

        # In[25]:

        prediction = model.predict(X_test, batch_size=1)

        from sklearn.metrics import classification_report

        prediction.shape
        print(classification_report(y_test.argmax(axis=2).flatten(), prediction.argmax(axis=2).flatten()))
        report_fname = '%s_%s_%s_%s_report.txt' % (NETWORK, WORK_ON, loss_name, dt_str)
        with open(report_fname, 'w') as fp:
            fp.write(classification_report(y_test.argmax(axis=2).flatten(), prediction.argmax(axis=2).flatten()))


        # In[26]:

        # Shows a random pred-true-image pair.
        # show_data(X_test,y_test,prediction)

        # In[33]:

        # Save compare pairs to folders
        def save_predictions(X, Y_true, Y_pred, path):
            # de-one-hot
            Y_true = np.array(Y_true.argmax(axis=2) * 255, dtype=np.uint8)
            Y_pred = np.array(Y_pred.argmax(axis=2) * 255, dtype=np.uint8)
            for n in range(len(X_test)):
                images, msk = X[n], Y_true[n]
                msk_pred = Y_pred[n]
                cv2.imwrite(path + str(n) + '_img.jpg', images[-1])
                cv2.imwrite(path + str(n) + '_true.jpg', msk.reshape(image_shape[0:2]))
                cv2.imwrite(path + str(n) + '_pred.jpg', msk_pred.reshape(image_shape[0:2]))


        os.makedirs('Dataset/result_pair_' + NETWORK + '_' + WORK_ON + '/' + dt_str + '/')
        save_predictions(X_test, y_test, prediction,
                         'Dataset/result_pair_' + NETWORK + '_' + WORK_ON + '/' + dt_str + '/')
        trained_weights = outfile = os.path.join('2DFCN2ECCV.h5')
        if os.path.exists(outfile):
            model.load_weights(trained_weights)
            print('Load trained weights')
        prediction = model.predict(X_test, batch_size=1)
        show_data(X_test, y_test, prediction)