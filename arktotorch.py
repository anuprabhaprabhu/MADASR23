
import torchaudio
import kaldiio
import torch

file ='/home/anuprabha/Documents/my_analysis_feats/feats/bh_D1_w2v_mdl/file.ark'
# d = { u:d for u,d in torchaudio.kaldi_io.read_vec_int_ark(file) }
# print(d)

data = kaldiio.load_ark(file)

# Convert data to tensors
tensor_dict = {}

# Convert data to tensors and store in the dictionary
for key, value in data:
    tensor_dict[key] = torch.tensor(value)  #.flatten()
    # print(torch.tensor(value).shape)
    break

# print(len(tensor_dict))
print(tensor_dict.value)


###############M    
### Note: It is better to apply CNN on features 
### CNN layers
# device = cpu
# def w2v_linear(feat,device):

#     wts = []

#     feat = torch.tensor(feat, device=device)
#     for i in range(len(feat)):

#         fc1 = nn.Linear(feat[i].shape[0],512)(feat[i])
#         fc2 = nn.Linear(fc1.shape[0],1)(fc1)
#         wts.extend(fc2)
    
#     with torch.no_grad():
#         op = wts
#     op_feats = op.cpu().numpy()

#     return op_feats


