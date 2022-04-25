from json import encoder
import os
import numpy as np
import sympy as sp
import torch

from utils import AttrDict
from envs import build_env
from models.transformer import TransformerModel

from utils import to_cuda
from envs.sympy_utils import simplify

model_path = './fwd_bwd_ibp.pth'

params = AttrDict({

    # environment parameters
    'env_name': 'char_sp',
    'int_base': 10,
    'balanced': False,
    'positive': True,
    'precision': 10,
    'n_variables': 1,
    'n_coefficients': 0,
    'leaf_probs': '0.75,0,0.25,0',
    'max_len': 512,
    'max_int': 5,
    'max_ops': 15,
    'max_ops_G': 15,
    'clean_prefix_expr': True,
    'rewrite_functions': '',
    'tasks': 'prim_ibp',
    'operators': 'add:10,sub:3,mul:10,div:5,sqrt:4,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4,tan:4,asin:1,acos:1,atan:1,sinh:1,cosh:1,tanh:1,asinh:1,acosh:1,atanh:1',

    # model parameters
    'cpu': True,
    'emb_dim': 1024,
    'n_enc_layers': 6,
    'n_dec_layers': 6,
    'n_heads': 8,
    'dropout': 0,
    'attention_dropout': 0,
    'sinusoidal_embeddings': False,
    'share_inout_emb': True,
    'reload_model': model_path,

    'reload_data': "prim_ibp,prim_ibp.train,prim_ibp.valid,prim_ibp.test",
    'reload_size': 1000,
    'batch_size': 4,
    'env_base_seed': 0,
    'num_workers': 10,
    'same_nb_ops_per_batch': False,

})
params.n_nodes = 1
params.node_id = 0
params.local_rank = 0
params.global_rank = 0
params.world_size = 1
params.n_gpu_per_node = 1

env = build_env(params)
x = env.local_dict['x']
print(x,'\n')

s = [x.split(',') for x in params.reload_data.split(';') if len(x) > 0]
data_path = {task: (train_path, valid_path, test_path) for task, train_path, valid_path, test_path in s}

dataloader = {
    task: iter(env.create_train_iterator(task, params, data_path))
    for task in params.tasks
}

dl = env.create_train_iterator(params.tasks[0], params, data_path)
# loader returns seq of length 3
# ((eq, eq_length), (sol, sol_length), nb_ops)

te = next(iter(dataloader))
print(te)

breakpoint()

F_infix = 'ln(cos(x + exp(x)) * sin(x**2 + 2) * exp(x) / x)'

F = sp.S(F_infix, locals=env.local_dict)
f = F.diff(x)
print('Equation:')
print(F,'\n')
print('Diffed eqn:')
print(f,'\n')

F_prefix = env.sympy_to_prefix(F)
f_prefix = env.sympy_to_prefix(f)
print(f"F prefix: {F_prefix}\n")
print(f"f prefix: {f_prefix}\n")


x1_prefix = env.clean_prefix(['sub', 'derivative', 'f', 'x', 'x'] + f_prefix)
print('')
print(x1_prefix)
print(len(x1_prefix),'\n')

x1 = torch.LongTensor(
    [env.eos_index] +
    [env.word2id[w] for w in x1_prefix] +
    [env.eos_index]
).view(-1, 1)
len1 = torch.LongTensor([len(x1)])
x1, len1 = to_cuda(x1, len1)

print('Input tensor:')
print(x1[:,0])
print(x1.shape)
print(len1, '\n')

enc_model = TransformerModel(params, env.id2word, is_encoder=True, with_output=False)
dec_model = TransformerModel(params, env.id2word, is_encoder=False, with_output=True)

with torch.no_grad():
    encoded = enc_model('fwd', x=x1, lengths=len1, causal=False).transpose(0, 1)

print("Encoder output:")
print(encoded)
print(encoded.shape,'\n')

beam_size = 10
with torch.no_grad():
    _, _, beam = dec_model.generate_beam(encoded, len1, beam_size=beam_size, length_penalty=1.0, early_stopping=1, max_len=200)
    assert len(beam) == 1
hypotheses = beam[0].hyp
assert len(hypotheses) == beam_size

print('Decoder output:')
print('  Beam length: ', len(beam))
print('  A Hypotheses: ', hypotheses[4])
print('  Hypotheses length: ', len(hypotheses))

# ids = hypotheses[4][1].tolist()
# tok = [env.id2word[wid] for wid in ids]
# print('Prefix hyp:')
# print(tok, '\n')

# infix = env.prefix_to_infix(tok)
# print('Infix hyp:')
# print(infix, '\n')

# sympy_hyp = env.infix_to_sympy(infix)
# print('Sympy hyp:')
# print(sympy_hyp, '\n')

print('\nNoise input to encoder:')
input_size = (134, 1)
noise_vec = (torch.rand(input_size)*20).round().type(torch.LongTensor)
noise_len = torch.LongTensor([len(noise_vec)])
print(noise_vec)
print(noise_vec.shape)
print(noise_len)

with torch.no_grad():
    enc_out_noise = enc_model('fwd', x=noise_vec, lengths=noise_len, causal=False).transpose(0, 1)

print(enc_out_noise)
print(enc_out_noise.shape)



