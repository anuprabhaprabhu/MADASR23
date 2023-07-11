import fairseq
import torch
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from kaldiio import WriteHelper

# python mad_feat_extrac.py /mnt/hdd1/MAD_ASR_hdd1/mad_trail/bn/train/D1/wav.scp /home/anuprabha/Documents/wav2vec_small.pt w2v
def load_my_mdl(path):

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([path])
    model = model[0]
    model.eval()
    # print(type(model[0]))
    # print(model.feature_extractor())
    return model



def load_my_audio(waveform, device):

    wave, sample_rate = torchaudio.load(waveform)
    wave = wave.numpy().astype(np.float32)
    wave = torch.tensor(wave, device = 'cpu')    
    return wave


def extractor(model,wave):

    with torch.no_grad():
        op = model.feature_extractor(wave)
    # op = torch.flatten(op[0])  # no flatten

    # op_feats = op.cpu().numpy()
    # print('op_feats type', type(op_feats) )
    # print('op_feats shape', op_feats.shape)
    # return op_feats
    return op


def write_kldfmt(wavscp_path, mdl_path, feats, device):
    
    data=pd.read_csv(wavscp_path,header=None,sep='\t')
    m1 = load_my_mdl(mdl_path) 
    

    # with WriteHelper('ark,scp:file.ark,file.scp') as writer:
    # extract feats for first elemeny in tensors
    print('start')
    feat =[]
    for i in data[1]:
        i = i.lstrip(' ')

        w1 = load_my_audio(i,device)
        w2v_raw = extractor(m1,w1)
        if feats == 'w2v':
            f1 = w2v_raw[0]

        # elif feats == 'lp_w2v':
        #     f1 = lp_w2v(i,w2v_raw[0])
        # elif feats == 'rwn':
        #     f1 = rwe_main(w2v_raw[0],device)
        else:
            print('Please select feature type')
        
        print('data type of feats',type(f1))
        print(i, f1.shape )
     
        feat.append([str(i), f1])   

        # writer(str(i),f1)
        # writer[i] = f1
        

    # write tensors in a file
    print(len(feat))
    path = '/mnt/hdd1/MAD_ASR_hdd1/mad_trail/bn/dev/D5/'
    torch.save(feat, path + 'feats.pt')
    
write_kldfmt(sys.argv[1], sys.argv[2], sys.argv[3], 'cpu')        ## wav.scp path, model path, w2v

# check = torch.load('feats.pt')
# print('check 0', check[0])
# print(type(check[0][1]))
# print('check 0 text', check[0][0])
# print('check 1 ',check[1])
