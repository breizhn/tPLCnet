"""
Script to run the pretrained TF-LITE models of the tPLCnet.
Just change "tflite_model_name" to the desired model and run the skript with:
$ python run_tPLCnet_tflite.py

Author: Nils L. Westhausen (Feb 2023)
"""

import tensorflow as tf
import os
import time
import numpy as np
from tqdm import tqdm
import soundfile as sf
import fnmatch
import scipy


os.environ["CUDA_VISIBLE_DEVICES"] = ''

path_to_audio = './test_files/'
tflite_model_name = './models/tPLCnet_l.tflite'
target_folder = './test_files_out/'

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# frame length for the model structure
frame_len = 160
print(tflite_model_name)
file_names = fnmatch.filter(os.listdir(path_to_audio), '*.wav')

interpreter_test = tf.lite.Interpreter(model_path=tflite_model_name)
interpreter_test.allocate_tensors()
input_details = interpreter_test.get_input_details()
output_details = interpreter_test.get_output_details()


for file in tqdm(file_names):
    
    # load audio file
    mix, fs = sf.read(os.path.join(path_to_audio, file))
    # load annotation for lost frames. It assumes annotation for 20 ms frames not 10 ms
    annotation = np.loadtxt(os.path.join(path_to_audio, file.replace('.wav','_is_lost.txt')))
    # pad the input file
    mix = np.concatenate((np.zeros((160)), mix, np.zeros((160))), axis=0)
    # double annotation since it is annotated for 20 ms.
    annotation = np.repeat(annotation, 2)
    # enable processing on the edges. Basically this tells the model to run one more time after a lost frame
    # to make a smooth transition to the original signal.
    annotation = annotation + np.roll(annotation, -1)
    annotation = (annotation > 0).astype('float32')
    annotation = np.concatenate((np.zeros((1)), annotation))
    # initialize buffers
    buffer = np.zeros(input_details[0]['shape']).astype('float32')
    last_out = np.zeros((1,1,160)).astype('float32')
    out_buffer = np.zeros((320)).astype('float32')
    win = scipy.signal.hann(320, sym=False)

    out_frames = []
    for idx, ano in enumerate(annotation):
        # fill buffer
        buffer = np.roll(buffer, -1, axis=1)
        buffer[0:1, -1, :] = np.copy(mix[(idx+1)*160:(idx+2)*160])
        buffer[0:1, -2, :] = np.copy(mix[(idx)*160:(idx+1)*160])
        buffer[0:1, -3, :] = np.copy(out_buffer[:160])
        
        if ano == 1:
            # run model if frame is lost
            interpreter_test.set_tensor(input_details[0]['index'], buffer.astype('float32'))
            interpreter_test.invoke()
            out_frame = interpreter_test.get_tensor(output_details[0]['index'])
            out_buffer = np.roll(out_buffer, -160)
            out_buffer[160:] = np.zeros((160))
            out_buffer = out_buffer + np.squeeze(out_frame) # hann window is compiled into the model
            out_frames.append(np.copy(out_buffer[:160]))

        else:
            # copy original signal if frame is not lost
            out_buffer = np.roll(out_buffer, -160)
            out_buffer[160:] = np.zeros((160))
            out_buffer = out_buffer + mix[(idx)*frame_len:(idx)*frame_len + 320] * win
            out_frames.append(np.copy(out_buffer[:160]))

    cleaned = np.reshape(np.stack(out_frames, axis=0), (-1))
    cleaned = np.squeeze(cleaned)
    out_audio = cleaned[160:]
    sf.write(os.path.join(target_folder, file), out_audio, fs)
    


