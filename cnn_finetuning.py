import argparse
import os
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras import backend as K

K.set_session(session)
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint

from params import data_root, results_root
from cnn_utils import Generator
from cnn_utils import custom_xception

import warnings

warnings.simplefilter('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--workers', type=int, default=cpu_count() // 2)
    parser.add_argument('--train_size', type=float, default=0.7)
    parser.add_argument('--train_compression', help='Apply random compression to training images', action='store_true',
                        default=False)
    parser.add_argument('--test_compression', help='Apply random compression to testing images', action='store_true',
                        default=False)
    parser.add_argument('--subsample', type=float, default=0.03)
    parser.add_argument('--leave_out_label', type=int)
    parser.add_argument('--recompression_qf', type=int)
    args = parser.parse_args()

    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    workers = args.workers
    train_size = args.train_size
    train_compression = args.train_compression
    test_compression = args.test_compression
    subsample = args.subsample
    leave_out_label = args.leave_out_label
    recompression_qf = args.recompression_qf

    np.random.seed(21)

    task_name = __file__.split('/')[-1].split('.')[0]
    print('TASK: {}'.format(task_name))

    recompression_qf_suf = '_{}'.format(recompression_qf)

    os.makedirs(os.path.join(results_root, task_name), exist_ok=True)

    # load model
    input_shape = (256, 256, 3)
    model = custom_xception(include_top=False, weights='imagenet', input_shape=input_shape, n_classes=2)

    log_file_train = os.path.join(results_root, task_name,
                                  'train_logo_{}_subsample_{}_train-compression_{}{}'
                                  '_test-compression_{}{}.csv'.format(leave_out_label,
                                                                      subsample,
                                                                      train_compression,
                                                                      recompression_qf_suf,
                                                                      test_compression,
                                                                      recompression_qf_suf))
    log_file_test = os.path.join(results_root, task_name,
                                 'test_logo_{}_subsample_{}_train-compression_{}{}'
                                 '_test-compression_{}{}.npy'.format(leave_out_label,
                                                                     subsample,
                                                                     train_compression,
                                                                     recompression_qf_suf,
                                                                     test_compression,
                                                                     recompression_qf_suf))

    if os.path.exists(log_file_test):
        print('\n\n\nLogo: {}, Subsample: {} already cmputed, skipping\n'.format(leave_out_label, subsample))
        return 0

    weights_path = os.path.join(results_root, task_name,
                                'best_weights_logo_{}_subsample_{}_train-compression_{}{}'
                                '_test-compression_{}{}.h5'.format(leave_out_label,
                                                                   subsample,
                                                                   train_compression,
                                                                   recompression_qf_suf,
                                                                   test_compression,
                                                                   recompression_qf_suf))

    csv_train_path = os.path.join(data_root,
                                  'logo_{}_split_train.csv'.format(leave_out_label))
    csv_test_path = os.path.join(data_root, 'logo_{}_split_test.csv'.format(leave_out_label))

    df_train_val = pd.read_csv(csv_train_path)
    df_train, df_val = train_test_split(df_train_val, train_size=train_size)
    df_test = pd.read_csv(csv_test_path)

    data_loader_train = Generator(df_train, patch_size=input_shape, compression=train_compression,
                                  batch_size=batch_size, subsample=subsample, recompression_qf=recompression_qf)
    data_loader_val = Generator(df_val, patch_size=input_shape, compression=train_compression,
                                batch_size=batch_size, subsample=subsample, recompression_qf=recompression_qf)
    data_loader_test = Generator(df_test, patch_size=input_shape, compression=test_compression,
                                 batch_size=batch_size, subsample=0.1, recompression_qf=recompression_qf)

    # compile the model
    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # finetune the model
    callbacks_train = [CSVLogger(log_file_train, separator=',', append=True),
                       ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                         patience=2, min_lr=1e-7),
                       EarlyStopping(monitor='val_loss', patience=3),
                       ModelCheckpoint(weights_path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
                       ]

    model.fit_generator(generator=data_loader_train, validation_data=data_loader_val, epochs=epochs,
                        callbacks=callbacks_train, workers=workers, use_multiprocessing=False,
                        verbose=0, max_queue_size=batch_size * 3)

    # test
    print('Loading best weights')

    model.load_weights(weights_path)

    print('\nTesting model')
    print('#' * 60)

    res = model.evaluate_generator(generator=data_loader_test, workers=workers, use_multiprocessing=True,
                                   verbose=0, max_queue_size=batch_size * 3)

    print('\n\n\nLogo: {}, Subsample: {}'.format(leave_out_label, subsample))
    print('Training samples: {}\nValidation samples: {}\nTesting samples: {}\n\n\n'.format(len(data_loader_train.db),
                                                                                           len(data_loader_val.db),
                                                                                           len(data_loader_test.db)))
    print('Test accuracy: {}'.format(res[1]))
    print('#' * 60)
    print('\n')

    np.save(log_file_test, {'test_acc': res[1]})

    return 0


if __name__ == '__main__':
    main()
