"""
Summary:  Train, inference and evaluate speech enhancement. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: -
"""
import numpy as np
import os
import pickle

import h5py
import argparse
import time
import glob
import matplotlib.pyplot as plt

import prepare_data as pp_data
import config as cfg
from spectrogram_to_wave import recover_wav

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.models import load_model
import tensorflow as tf
import csv


n_concat = cfg.n_concat
n_freq = cfg.n_freq

def eval(model, gen, x, y):
    """Validation function. 
    
    Args:
      model: keras model. 
      gen: object, data generator. 
      x: 3darray, input, (n_segs, n_concat, n_freq)
      y: 2darray, target, (n_segs, n_freq)
    """
    pred_all, y_all = [], []
    
    # Inference in mini batch. 
    for (batch_x, batch_y) in gen.generate(xs=[x], ys=[y]):
        pred = model.predict(batch_x)
        pred_all.append(pred)
        y_all.append(batch_y)
        
    # Concatenate mini batch prediction. 
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    # Compute loss. 
    loss = pp_data.np_mean_absolute_error(y_all, pred_all)
    return loss
    
    """Why not use model.evaluate() directly--Yb
    """
def mask_mse_loss(clean_and_mix, mask):
    clean = clean_and_mix[:,:n_freq]
    mix = clean_and_mix[:,n_freq:]
    return(tf.reduce_sum(tf.square(tf.multiply(mix, mask) - clean)))
    # return tf.keras.losses.MSE(clean, tf.multiply(mix, mask))
    
def create_model():
    n_hid = tf.constant(1024)
    
    # define loss function
    model = Sequential()
    model.add(Dense(input_shape=(n_concat*n_freq,), units=n_hid, activation='relu'))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dense(n_hid, activation='relu'))
    model.add(Dense(n_freq, activation='sigmoid'))
    # model.compile(loss=mask_mse_loss, optimizer=Adagrad(learning_rate=lr_schedule))
    return model
        
def train(args):
    """Train the neural network. Write out model every several iterations. 
    
    Args:
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      te_snr: float, testing SNR. 
      lr: float, learning rate. 
    """
    print(args)
    workspace = args.workspace
    lr_initial = args.lr
    batchsize = 512
    epochs = 1
    
    # 1.1 Load train dataset
    tr_path = os.path.join(workspace, "3_packed_features", "spectrogram", "train","*.tfrecord")
    tr_ds = pp_data.load_and_parse_tfrecord(tr_path, batch_size=batchsize, file_subset=-1)
    
    # 1.2 Load test subset for validation
    te_path = os.path.join(workspace, "3_packed_features", "spectrogram", "test", "*.tfrecord")
    te_ds = pp_data.load_and_parse_tfrecord(te_path, batch_size=batchsize)
    
    # 2. Build model
    lr_schedule=tf.keras.optimizers.schedules.ExponentialDecay(lr_initial, decay_steps=100000,decay_rate=0.5,staircase=True)
    distribute_strategy = tf.distribute.MirroredStrategy()#devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"]
    with distribute_strategy.scope():
        model = create_model()
        model.compile(loss=mask_mse_loss, optimizer=Adagrad(learning_rate=lr_schedule))
    # # model.summary()    

    
    # 3. Directories for saving models
    model_dir = os.path.join(workspace, "4_models")
    pp_data.create_folder(model_dir)
    # model_path = os.path.join(model_dir, "weights.03-10.3224")
    # tf.saved_model.load(model_path)

    
    # 5. Train. 
    
    # define callbacks
    model_name = os.path.join(model_dir, "weights.{epoch:02d}-{loss:.4f}.hdf5")
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=model_name, save_best_only=False, save_weights_only=False, save_freq='epoch')]
    
    t1 = time.time()
    model.fit(tr_ds, validation_data=te_ds, epochs=3, callbacks=callbacks)
    print("Training time: %s s" % (time.time() - t1,))



def inference(args):
    """Inference all test data, write out recovered wavs to disk. 
    
    Args:
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      te_snr: float, testing SNR. 
      n_concat: int, number of frames to concatenta, should equal to n_concat 
          in the training stage. 
      iter: int, iteration of model to load. 
      visualize: bool, plot enhanced spectrogram for debug. 
    """
    print(args)
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    n_concat = cfg.n_concat
    # iter = args.iteration
    
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    scale = False
    
    # Load model. 
    model_path = os.path.join(workspace, "4_models", "weights.03-97.0369.hdf5")
    model = create_model()
    model.load_weights(model_path)
    # model = load_model(model_path)

    # Load test data. 
    feat_dir = os.path.join(workspace, "2_features", "spectrogram", "test", "%ddb" % int(te_snr))
    names = os.listdir(feat_dir)

    for (cnt, na) in enumerate(names):
        # Load feature. 
        feat_path = os.path.join(feat_dir, na)
        data = pickle.load(open(feat_path, 'rb'))
        [mixed_cmplx_x, speech_x, noise_x, alpha, na] = data
        mixed_x = np.abs(mixed_cmplx_x)
        
        # Process data. 
        n_pad = (n_concat - 1) // 2
        mixed_x = pp_data.pad_with_border(mixed_x, n_pad)

        
        # Scale data. 
        if scale:
            mixed_x = pp_data.scale_on_2d(mixed_x, scaler)
            speech_x = pp_data.scale_on_2d(speech_x, scaler)
        
        # Cut input spectrogram to 3D segments with n_concat. 
        mixed_x_3d = pp_data.mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)
        
        # Predict. 
        def predict(model, mixed_x, n_centr):
            pred_logsp = np.zeros_like(mixed_x[:,1,:])
            for n_seg in range(mixed_x.shape[0]):
                # predict mask
                input = mixed_x[n_seg,:,:].reshape(1,-1)
                pred_mask = model(input)
                # apply mask on mix
                pred_logsp[n_seg,:] = np.multiply(pred_mask, mixed_x[n_seg,n_centr,:])
            return(pred_logsp)
        pred_mask = model.predict(mixed_x_3d.reshape(-1,n_concat*n_freq))
        
        print(cnt, na)

               
        # Debug plot. 
        if args.visualize:
            fig, axs = plt.subplots(3,1, sharex=False)
            
            mixed_x = pp_data.log_sp(mixed_x)
            speech_x = pp_data.log_sp(speech_x)
            
            axs[0].matshow(mixed_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[1].matshow(speech_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[2].matshow(pred_mask.T, origin='lower', aspect='auto', cmap='jet')
            axs[0].set_title("%ddb mixture log spectrogram" % int(te_snr))
            axs[1].set_title("Clean speech log spectrogram")
            axs[2].set_title("Enhanced speech log spectrogram")
            for j1 in range(3):
                axs[j1].xaxis.tick_bottom()
            plt.tight_layout()
            # plt.show()
            plt.draw()
            plt.savefig(os.path.join(workspace, "%s_LogSP_and_Mask.png" % na))

        # Recover enhanced wav. 
        pred_sp = tf.multiply(pred_mask, np.abs(mixed_cmplx_x))
        s = recover_wav(pred_sp, mixed_cmplx_x, n_overlap, np.hamming)
        s *= np.sqrt((np.hamming(n_window)**2).sum())   # Scaler for compensate the amplitude 
                                                        # change after spectrogram and IFFT. 
        
        # Write out enhanced wav. 
        out_path = os.path.join(workspace, "5_enh_wavs", "test", "%ddb" % int(te_snr), "%s.enh.wav" % na)
        pp_data.create_folder(os.path.dirname(out_path))
        pp_data.write_audio(out_path, s, fs)
        
        x = recover_wav(np.abs(mixed_cmplx_x), mixed_cmplx_x, n_overlap, np.hamming)
        x *= np.sqrt((np.hamming(n_window)**2).sum())
        pp_data.write_audio(os.path.join(workspace, "5_enh_wavs", "test", "%ddb" % int(te_snr), "%s.wav" % na), x, fs)
        
        if cnt == 1:
            break
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--lr', type=float, required=True)
    
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--tr_snr', type=float, required=True)
    parser_inference.add_argument('--te_snr', type=float, required=True)
    # parser_inference.add_argument('--n_concat', type=int, required=True)
    # parser_inference.add_argument('--iteration', type=int, required=True)
    parser_inference.add_argument('--visualize', action='store_true', default=False)
    
    parser_calculate_pesq = subparsers.add_parser('calculate_pesq')
    parser_calculate_pesq.add_argument('--workspace', type=str, required=True)
    parser_calculate_pesq.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_pesq.add_argument('--te_snr', type=float, required=True)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)
    elif args.mode == 'calculate_pesq':
        calculate_pesq(args)
    else:
        raise Exception("Error!")