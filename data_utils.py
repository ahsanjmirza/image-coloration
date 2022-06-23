import os
import numpy as np
import random
import pandas as pd

def train_data_generator(dataset_path, continue_flag):
    steps_per_epoch = get_steps_per_epoch(dataset_path)
    save_flag = 0
    if continue_flag == 'True':
        datalist = pd.read_csv(os.path.join('./', 'train_logs.csv'))['Unprocessed'].values.tolist()
        print('Left:', len(datalist))
    else: 
        datalist = os.listdir(os.path.join(dataset_path, 'color'))
        random.shuffle(datalist)
        df = pd.DataFrame({'Unprocessed': datalist})
        df.to_csv(os.path.join('./', 'train_logs.csv'), index=False) 
    while(1):
        random.shuffle(datalist)
        while datalist:
            f = datalist.pop()
            if f.endswith('.npy'):
                batch_0 = np.float32(np.load(os.path.join(dataset_path, 'grey', f)))
                batch_1 = np.float32(np.load(os.path.join(dataset_path, 'color', f)))
                yield (batch_0, batch_1)
                del batch_0, batch_1
                save_flag += 1
            if steps_per_epoch - save_flag == 0:
                df = pd.DataFrame({'Unprocessed': datalist})
                df.to_csv(os.path.join('./', 'train_logs.csv'), index=False) 
                save_flag = 0
        datalist = os.listdir(dataset_path)

def get_steps_per_epoch(dataset_path):
    return int(len(os.listdir(os.path.join(dataset_path, 'color'))))