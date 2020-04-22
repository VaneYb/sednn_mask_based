"""
Summary:  Prepare data. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified 
--By Yb on 2020.3.20
    Add shuffle_and_pack_features function
--By Yb on 2020.4.16
    ** Add read and write functions of tfrecord file for large training dataset
    ** Update packed_feature, for mixed_central_logSp is used in the mask method for loss computation
"""
import os
import sys
# print(sys.version)
import soundfile
import numpy as np
import argparse
import csv
import time
import matplotlib.pyplot as plt
from scipy import signal
import h5py
from sklearn import preprocessing
import librosa
import prepare_data as pp_data
import config as cfg
import tensorflow as tf
# print(tf.__version__)
# print(tf.keras.__version__)

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs
    
def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

###
def create_mixture_csv(args):
    """Create csv containing mixture information. 
    Each line in the .csv file contains [speech_name, noise_name, noise_onset, noise_offset]
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of speech data. 
      noise_dir: str, path of noise data. 
      data_type: str, 'train' | 'test'. 
      magnification: int, only used when data_type='train', number of noise 
          selected to mix with a speech. E.g., when magnication=3, then 4620
          speech with create 4620*3 mixtures. magnification should not larger 
          than the species of noises. 
    """
    workspace = args.workspace
    speech_dir = args.speech_dir
    noise_dir = args.noise_dir
    data_type = args.data_type
    magnification = args.magnification
    fs = cfg.sample_rate
    
    speech_names = [na for na in os.listdir(speech_dir) if na.lower().endswith(".wav")]
    noise_names = [na for na in os.listdir(noise_dir) if na.lower().endswith(".wav")]
    
    rs = np.random.RandomState(0)
    out_csv_path = os.path.join(workspace, "1_mixture_csvs", "%s.csv" % data_type)
    pp_data.create_folder(os.path.dirname(out_csv_path))
    
    cnt = 0
    f = open(out_csv_path, 'w')
    f.write("%s\t%s\t%s\t%s\n" % ("speech_name", "noise_name", "noise_onset", "noise_offset"))
    for speech_na in speech_names:
        # Read speech. 
        speech_path = os.path.join(speech_dir, speech_na)
        (speech_audio, _) = read_audio(speech_path, target_fs=fs)
        len_speech = len(speech_audio)
        
        # For training data, mix each speech with randomly picked #magnification noises. 
        if data_type == 'train':
            selected_noise_names = rs.choice(noise_names, size=magnification, replace=False)
        # For test data, mix each speech with all noises. 
        elif data_type == 'test':
            selected_noise_names = noise_names
        else:
            raise Exception("data_type must be train | test!")

        # Mix one speech with different noises many times. 
        for noise_na in selected_noise_names:
            noise_path = os.path.join(noise_dir, noise_na)
            (noise_audio, _) = read_audio(noise_path, target_fs=fs)
            
            len_noise = len(noise_audio)

            if len_noise <= len_speech:
                noise_onset = 0
                noise_offset = len_speech
            # If noise longer than speech then randomly select a segment of noise. 
            else:
                noise_onset = rs.randint(0, len_noise - len_speech, size=1)[0]
                noise_offset = noise_onset + len_speech
            
            if cnt % 100 == 0:
                print(cnt)
                
            f.write("%s\t%s\t%d\t%d\n" % (speech_na, noise_na, noise_onset, noise_offset))
            
            cnt += 1
               
    f.close()
    print(out_csv_path)
    print("Create %s mixture csv finished!" % data_type)
    
###
def calculate_mixture_features(args):
    """Calculate spectrogram for mixed, speech and noise audio. Then write the 
    features to disk. 
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of speech data. 
      noise_dir: str, path of noise data. 
      data_type: str, 'train' | 'test'. 
      snr: float, signal to noise ratio to be mixed. 
    """
    workspace = args.workspace
    speech_dir = args.speech_dir
    noise_dir = args.noise_dir
    data_type = args.data_type
    snr = args.snr
    fs = cfg.sample_rate
    
    # Open mixture csv. 
    mixture_csv_path = os.path.join(workspace, "1_mixture_csvs", "%s.csv" % data_type)
    with open(mixture_csv_path, 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)
    
    t1 = time.time()
    cnt = 0
    for i1 in range(1, len(lis)):
        [speech_na, noise_na, noise_onset, noise_offset] = lis[i1]
        noise_onset = int(noise_onset)
        noise_offset = int(noise_offset)
        
        # Read speech audio. 
        speech_path = os.path.join(speech_dir, speech_na)
        (speech_audio, _) = read_audio(speech_path, target_fs=fs)
        
        # Read noise audio. 
        noise_path = os.path.join(noise_dir, noise_na)
        (noise_audio, _) = read_audio(noise_path, target_fs=fs)
        
        # Repeat noise to the same length as speech. 
        if len(noise_audio) < len(speech_audio):
            n_repeat = int(np.ceil(float(len(speech_audio)) / float(len(noise_audio))))
            noise_audio_ex = np.tile(noise_audio, n_repeat)
            noise_audio = noise_audio_ex[0 : len(speech_audio)]
        # Truncate noise to the same length as speech. 
        else:
            noise_audio = noise_audio[noise_onset : noise_offset]
        
        # Scale speech to given snr. 
        scaler = get_amplitude_scaling_factor(speech_audio, noise_audio, snr=snr)
        speech_audio *= scaler
        
        # Get normalized mixture, speech, noise. 
        (mixed_audio, speech_audio, noise_audio, alpha) = additive_mixing(speech_audio, noise_audio)

        # Write out mixed audio. 
        out_bare_na = os.path.join("%s.%s" % 
            (os.path.splitext(speech_na)[0], os.path.splitext(noise_na)[0]))
        out_audio_path = os.path.join(workspace, "2_mixed_audios", "spectrogram", 
            data_type, "%ddb" % int(snr), "%s.wav" % out_bare_na)
        create_folder(os.path.dirname(out_audio_path))
        write_audio(out_audio_path, mixed_audio, fs)

        # Extract spectrogram. 
        mixed_complx_x = calc_sp(mixed_audio, mode='complex')
        speech_x = calc_sp(speech_audio, mode='magnitude')
        noise_x = calc_sp(noise_audio, mode='magnitude')

        # Write out features. 
        out_feat_path = os.path.join(workspace, "2_features", "spectrogram", 
            data_type, "%ddb" % int(snr), "%s.p" % out_bare_na)
        create_folder(os.path.dirname(out_feat_path))
        data = [mixed_complx_x, speech_x, noise_x, alpha, out_bare_na]
        pickle.dump(data, open(out_feat_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        # Print. 
        if cnt % 100 == 0:
            print(cnt)
            
        cnt += 1

    print("Extracting feature time: %s" % (time.time() - t1), "SNR = %d" % snr)
    
def rms(y):
    """Root mean square. 
    """
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))

def get_amplitude_scaling_factor(s, n, snr, method='rms'):
    """Given s and n, return the scaler s according to the snr. 
    
    Args:
      s: ndarray, source1. 
      n: ndarray, source2. 
      snr: float, SNR. 
      method: 'rms'. 
      
    Outputs:
      float, scaler. 
    """
    original_sn_rms_ratio = rms(s) / rms(n)
    target_sn_rms_ratio =  10. ** (float(snr) / 20.)    # snr = 20 * lg(rms(s) / rms(n))
    signal_scaling_factor = target_sn_rms_ratio / original_sn_rms_ratio
    return signal_scaling_factor

def additive_mixing(s, n):
    """Mix normalized source1 and source2. 
    
    Args:
      s: ndarray, source1. 
      n: ndarray, source2. 
      
    Returns:
      mix_audio: ndarray, mixed audio. 
      s: ndarray, pad or truncated and scalered source1. 
      n: ndarray, scaled source2. 
      alpha: float, normalize coefficient. 
    """
    mixed_audio = s + n
        
    alpha = 1. / np.max(np.abs(mixed_audio))
    mixed_audio *= alpha
    s *= alpha
    n *= alpha
    return mixed_audio, s, n, alpha
    
def calc_sp(audio, mode):
    """Calculate spectrogram. 
    
    Args:
      audio: 1darray. 
      mode: string, 'magnitude' | 'complex'
    
    Returns:
      spectrogram: 2darray, (n_time, n_freq). 
    """
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    ham_win = np.hamming(n_window)
    [f, t, x] = signal.spectral.spectrogram(
                    audio, 
                    window=ham_win,
                    nperseg=n_window, 
                    noverlap=n_overlap, 
                    detrend=False, 
                    return_onesided=True, 
                    mode=mode) 
    x = x.T
    if mode == 'magnitude':
        x = x.astype(np.float32)
    elif mode == 'complex':
        x = x.astype(np.complex64)
    else:
        raise Exception("Incorrect mode!")
    return x
    
def write_to_tfrecord(filename, mix_feature, clean_feature):
    """ Save features to a TFRecord file.
    
    Features:
        input: mix_feature(n_segs, n_concat, n_freq)
        clean_and_mix: concatenate(
                            clean_feature(n_segs, n_freq),
                            mix_feature(n_segs, (n_concat-1)/2, n_freq))
    """

    with tf.io.TFRecordWriter(filename) as writer:
        for i_eg in range(mix_feature.shape[0]):
        
            # reshape array to list
            mix = mix_feature[i_eg, :, :].reshape(1,-1).tolist()[0]
            
            # concatenate clean with central mix_feature used to compute loss when training
            frames = np.concatenate((clean_feature[i_eg, :], mix_feature[i_eg, 2, :].reshape(-1,)), axis=0)


            # convert each observation to a tf.Example message
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'mix': tf.train.Feature(float_list=tf.train.FloatList(value=mix)),
                        'clean_and_mix_frame': tf.train.Feature(float_list=tf.train.FloatList(value=frames))}))
            
            # written to file
            writer.write(example.SerializeToString())

def _parse_function(example_proto):
    return tf.io.parse_example(example_proto, features = {
        'mix': tf.io.FixedLenFeature([cfg.n_concat*cfg.n_freq], tf.float32),
        'clean_and_mix_frame': tf.io.FixedLenFeature([2*cfg.n_freq], tf.float32)   
    })


def load_and_parse_tfrecord(tfrecord_path, batch_size=1, file_subset = -1):
    files = tf.data.Dataset.list_files(tfrecord_path, shuffle=True, seed=1) 
    if (file_subset < 0): 
        dataset = files.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = files.take(file_subset).interleave(tf.data.TFRecordDataset, cycle_length=32)
    return dataset.batch(batch_size).map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).map(lambda x:(x['mix'], x['clean_and_mix_frame']))
                

def shuffle_and_pack_features(args):
    """1. Shuffle the snr and order of the name list in each snr.
       2. Load and pack the features of each mixture and of its corresponding clean audio.
       Apply log and conver to 3D tensor, write all the frame features out to .h5 file along each mixture.
    
    Args:
      workspace: str, path of workspace. 
      data_type: str, 'train' | 'test'. 
    """
    workspace = args.workspace
    data_type = args.data_type
    # snr = args.snr
    n_concat = cfg.n_concat
    n_hop = cfg.n_hop
    
    # Create packed feature outpath
    out_path = os.path.join(workspace, "3_packed_features", "spectrogram", data_type)
    create_folder(out_path)
    
    # Read in snr list and wav feature list
    feat_path = os.path.join(workspace, "2_features", "spectrogram", data_type)
    snr_list = os.listdir(feat_path)
    n_snr = len(snr_list)
    
    feat_dir = os.path.join(feat_path, snr_list[0])
    feat_list = os.listdir(feat_dir)
    n_feat = len(feat_list)
    
    totCnt = n_feat * n_snr
    
    # Initialize shuffled index of queues
    snr_queue = np.arange(n_snr).reshape(-1,1) * np.ones(shape=(1,n_feat), dtype=int)
    snr_queue = snr_queue.reshape(1,-1).squeeze()
    np.random.shuffle(snr_queue)
    snr_queue = list(snr_queue)
    
    feat_queue = []
    idlist = np.arange(n_feat)
    for i1 in range(n_snr):
        np.random.shuffle(idlist)
        tmp = list(idlist)
        feat_queue.append(tmp)
    
    # Start packing features
    x_all = []  # (n_segs, n_concat, n_freq)
    y_all = []  # (n_segs, n_freq)
    
    cnt = 0
    t1 = time.time()
    while(True):
        snr_id = snr_queue.pop(0)
        snr_type = snr_list[snr_id]
        
        fea_id = feat_queue[snr_id].pop(0)
        fea_na = feat_list[fea_id]
        
        # Load feature. 
        feat_path = os.path.join(workspace, "2_features", "spectrogram", data_type, snr_type, fea_na)
        data = pickle.load(open(feat_path, 'rb'))
                
        [mixed_complx_x, speech_x, noise_x, alpha, na] = data   # (620,257)
        mixed_x = np.abs(mixed_complx_x)      
        # print("Read in,",mixed_x.shape, speech_x.shape)
        
        # Pad start and finish of the spectrogram with boarder values. 
        n_pad = (n_concat - 1) // 2
        mixed_x = pad_with_border(mixed_x, n_pad)
        speech_x = pad_with_border(speech_x, n_pad)     #(488,257)
        # print("After pad,",mixed_x.shape, speech_x.shape)
        
        # Cut input spectrogram to 3D segments with n_concat. 
        mixed_x_3d = mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=n_hop)
        x_all.append(mixed_x_3d)
        # print("To 3D,",mixed_x_3d.shape)
        
        # Cut target spectrogram and take the center frame of each 3D segment. 
        speech_x_3d = mat_2d_to_3d(speech_x, agg_num=n_concat, hop=n_hop)
        y = speech_x_3d[:, (n_concat - 1) // 2, :]
        y_all.append(y)
        # print("To 3D,",y.shape)
        
        cnt += 1
        if (cnt%1000 == 0 or cnt==totCnt):
        
            x_all = np.concatenate(x_all, axis=0).astype(np.float32)   # (n_segs, n_concat, n_freq)
            y_all = np.concatenate(y_all, axis=0).astype(np.float32)   # (n_segs, n_freq)
    
            # x_all = log_sp(x_all).astype(np.float32)
            # y_all = log_sp(y_all).astype(np.float32)
            
            """
            # Write data out to .h5 file. 
            if (cnt==totCnt):
                out_path = os.path.join(workspace, "3_packed_features", "spectrogram", data_type, "data%d.h5" % (cnt//1000+1))
            else:
                out_path = os.path.join(workspace, "3_packed_features", "spectrogram", data_type, "data%d.h5" % (cnt//1000))
            
            create_folder(os.path.dirname(out_path))
            with h5py.File(out_path, 'w') as hf:
                hf.create_dataset('x', data=x_all)
                hf.create_dataset('y', data=y_all)
            """
            # Write data out to tfrecord
            if (cnt==totCnt):
                out_name = os.path.join(out_path, "data%d.tfrecord" % (cnt//1000))
            else:
                out_name = os.path.join(out_path, "data%d.tfrecord" % (cnt//1000-1))
            write_to_tfrecord(filename=out_name, mix_feature=x_all, clean_feature=y_all)
            
            print("Writing pack features out to %s" % out_name, "(%d/%d)" % (cnt, totCnt))
            
            x_all = []
            y_all = []
        
        if (cnt == totCnt):
            break
       
    print("Pack features finished! %s s" % (time.time() - t1,))

    
def pack_features(args):
    """Load features, apply log and conver to 3D tensor, only write packed (features & targets) out to tfrecord file. 
    
    Args:
      workspace: str, path of workspace. 
      data_type: str, 'train' | 'test'. 
      snr: float, signal to noise ratio to be mixed. 
      n_concat: int, number of frames to be concatenated. 
      n_hop: int, hop frames. 
    """
    workspace = args.workspace
    data_type = args.data_type
    snr = args.snr
    n_concat = args.n_concat
    n_hop = args.n_hop
    
    x_all = []  # (n_segs, n_concat, n_freq)
    y_all = []  # (n_segs, n_freq)
    
    cnt = 0
    t1 = time.time()
    
    # Load features. 
    feat_dir = os.path.join(workspace, "3_features", "spectrogram", data_type, "%ddb" % int(snr))
    names = os.listdir(feat_dir)
    
    np.random.shuffle(names)
    for na in names:
        # Load feature. 
        feat_path = os.path.join(feat_dir, na)
        data = pickle.load(open(feat_path, 'rb'))
        [mixed_complx_x, speech_x, noise_x, alpha, na] = data
        mixed_x = np.abs(mixed_complx_x)

        # Pad start and finish of the spectrogram with boarder values. 
        n_pad = (n_concat - 1) // 2
        mixed_x = pad_with_border(mixed_x, n_pad)
        speech_x = pad_with_border(speech_x, n_pad)
    
        # Cut input spectrogram to 3D segments with n_concat. 
        mixed_x_3d = mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=n_hop)
        x_all.append(mixed_x_3d)
        
        # Cut target spectrogram and take the center frame of each 3D segment. 
        speech_x_3d = mat_2d_to_3d(speech_x, agg_num=n_concat, hop=n_hop)
        y = speech_x_3d[:, (n_concat - 1) // 2, :]
        y_all.append(y)
    
        # Print. 
        if cnt % 100 == 0:
            print(cnt)
            
        if cnt == 3: break
        cnt += 1
    
    x_all = np.concatenate(x_all, axis=0)   # (n_segs, n_concat, n_freq)
    y_all = np.concatenate(y_all, axis=0)   # (n_segs, n_freq)
    
    x_all = log_sp(x_all).astype(np.float32)
    y_all = log_sp(y_all).astype(np.float32)
    
    # Write out data to .h5 file. 
    out_path = os.path.join(workspace, "packed_features", "spectrogram", data_type, "%ddb" % int(snr), "data.h5")
    
    print("Writing pack features out to %s" % out_path)
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('x', data=x_all)
        hf.create_dataset('y', data=y_all)
        
    print("Pack features finished! %s s" % (time.time() - t1,))
    
def log_sp(x):
    return np.log(x + 1e-08)
    
def mat_2d_to_3d(x, agg_num, hop):
    """Segment 2D array to 3D segments. 
    """
    # Pad to at least one block. 
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))
        
    # Segment 2d to 3d. 
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1 : i1 + agg_num])
        i1 += hop
    return np.array(x3d)

def pad_with_border(x, n_pad):
    """Pad the begin and finish of spectrogram with border frame value. 
    """
    x_pad_list = [x[0:1]] * n_pad + [x] + [x[-1:]] * n_pad
    return np.concatenate(x_pad_list, axis=0)
    """ replacing this part by extend()--Yb
    """

###
def compute_scaler(args):
    """Compute and write out scaler of data. 
    """
    workspace = args.workspace
    data_type = args.data_type
    snr = args.snr
    
    # Load data. 
    t1 = time.time()
    hdf5_path = os.path.join(workspace, "4_packed_features", "spectrogram", data_type, "%ddb" % int(snr), "data.h5")
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')     
        x = np.array(x)     # (n_segs, n_concat, n_freq)
    
    # Compute scaler. 
    (n_segs, n_concat, n_freq) = x.shape
    x2d = x.reshape((n_segs * n_concat, n_freq))
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(x2d)
    # print(scaler.mean_)
    # print(scaler.scale_)
    
    # Write out scaler. 
    out_path = os.path.join(workspace, "4_packed_features", "spectrogram", data_type, "%ddb" % int(snr), "scaler.p")
    create_folder(os.path.dirname(out_path))
    pickle.dump(scaler, open(out_path, 'wb'))
    
    print("Save scaler to %s" % out_path)
    print("Compute scaler finished! %s s" % (time.time() - t1,))
    
def scale_on_2d(x2d, scaler):
    """Scale 2D array data. 
    """
    return scaler.transform(x2d)
    
def scale_on_3d(x3d, scaler):
    """Scale 3D array data. 
    """
    (n_segs, n_concat, n_freq) = x3d.shape
    x2d = x3d.reshape((n_segs * n_concat, n_freq))
    x2d = scaler.transform(x2d)
    x3d = x2d.reshape((n_segs, n_concat, n_freq))
    return x3d
    
def inverse_scale_on_2d(x2d, scaler):
    """Inverse scale 2D array data. 
    """
    return x2d * scaler.scale_[None, :] + scaler.mean_[None, :]
    
###
def load_hdf5(hdf5_path):
    """Load hdf5 data. 
    """
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        x = np.array(x)     # (n_segs, n_concat, n_freq)
        y = np.array(y)     # (n_segs, n_freq)        
    return x, y

def np_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))
    
###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_create_mixture_csv = subparsers.add_parser('create_mixture_csv')
    parser_create_mixture_csv.add_argument('--workspace', type=str, required=True)
    parser_create_mixture_csv.add_argument('--speech_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--noise_dir', type=str, required=True)
    parser_create_mixture_csv.add_argument('--data_type', type=str, required=True)
    parser_create_mixture_csv.add_argument('--magnification', type=int, default=1)

    parser_calculate_mixture_features = subparsers.add_parser('calculate_mixture_features')
    parser_calculate_mixture_features.add_argument('--workspace', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--noise_dir', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--data_type', type=str, required=True)
    parser_calculate_mixture_features.add_argument('--snr', type=float, required=True)
    
    parser_pack_features = subparsers.add_parser('pack_features')
    parser_pack_features.add_argument('--workspace', type=str, required=True)
    parser_pack_features.add_argument('--data_type', type=str, required=True)
    parser_pack_features.add_argument('--snr', type=float, required=True)
    parser_pack_features.add_argument('--n_concat', type=int, required=True)
    parser_pack_features.add_argument('--n_hop', type=int, required=True)
    
    parser_compute_scaler = subparsers.add_parser('compute_scaler')
    parser_compute_scaler.add_argument('--workspace', type=str, required=True)
    parser_compute_scaler.add_argument('--data_type', type=str, required=True)
    parser_compute_scaler.add_argument('--snr', type=float, required=True)
    
    parser_pack_features = subparsers.add_parser('shuffle_and_pack_features')
    parser_pack_features.add_argument('--workspace', type=str, required=True)
    parser_pack_features.add_argument('--data_type', type=str, required=True)
    # parser_pack_features.add_argument('--snr', type=float, required=True)
    # parser_pack_features.add_argument('--n_concat', type=int, required=True)
    # parser_pack_features.add_argument('--n_hop', type=int, required=True)
    
    args = parser.parse_args()
    if args.mode == 'create_mixture_csv':
        create_mixture_csv(args)
    elif args.mode == 'calculate_mixture_features':
        calculate_mixture_features(args)
    elif args.mode == 'pack_features':
        pack_features(args)       
    elif args.mode == 'compute_scaler':
        compute_scaler(args)
    elif args.mode == 'shuffle_and_pack_features':
        shuffle_and_pack_features(args) 
    else:
        raise Exception("Error!")
