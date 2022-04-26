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

import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# sys.stdout = open("results/console_output_fixedexpressions.txt", "w")

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

    # Train params
    # I'm not 100% sure what all these do, but they're all required for the model to train
    'reload_data': "prim_ibp,prim_ibp.train,prim_ibp.valid,prim_ibp.test",
    'reload_size': 1000,
    'batch_size': 4,
    'env_base_seed': 0,
    'n_nodes':1,
    'node_id':0,
    'local_rank':0,
    'global_rank':0,
    'world_size':1,
    'n_gpu_per_node':1,
    'num_workers': 10,
    'same_nb_ops_per_batch': False,


})

# Train one batch
def enc_dec_step(batch, optimizer, encoder, decoder):
        """
        Encoding / decoding step.
        """
        encoder.train()
        decoder.train()

        # unpack batch
        # x1 = equation to solve, x2 = equation solution
        # Hence, we only care about x1 for equation generation
        # TODO: will want to stop x2 from loading at all to save memory
        (x1, len1), (x2, len2), _ = batch

        # Start of equation is the problem type (eg. -Y')
        # TODO: generalise to other problem types
        eq_start = x1[:3]
        eq_start_len = torch.clone(len1)
        eq_start_len[:] = 3

        # Output should be the whole equation
        eq_out = x1
        eq_out_len = len1


        # target words to predict
        alen = torch.arange(eq_out_len.max(), dtype=torch.long, device=eq_out_len.device)
        pred_mask = alen[:, None] < eq_out_len[None] - 1  # do not predict anything given the last target word
        y = eq_out[1:].masked_select(pred_mask[:-1])
        assert len(y) == (eq_out_len - 1).sum().item()

        # cuda
        eq_start, eq_start_len, eq_out, eq_out_len, y = to_cuda(eq_start, eq_start_len, eq_out, eq_out_len, y)

        # forward pass
        encoded = encoder('fwd', x=eq_start, lengths=eq_start_len, causal=False)
        decoded = decoder('fwd', x=eq_out, lengths=eq_out_len, causal=True, src_enc=encoded.transpose(0, 1), src_len=eq_start_len)
        
        # Prediction step to get loss
        _, loss = decoder('predict', tensor=decoded, pred_mask=pred_mask, y=y, get_scores=False)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss


# Generate some problems using the transformer
def test_seq_gen(env, enc_model, dec_model):

    # This is the input for integral problems (-Y')
    test_seq = torch.LongTensor([0, 67, 79]).view(-1, 1)
    test_len = torch.LongTensor([3])

    # Fwd encoder pass
    with torch.no_grad():
        encoded = enc_model('fwd', x=test_seq, lengths=test_len, causal=False).transpose(0, 1)

    # Do a beam search to generate outputs
    beam_size = 10
    with torch.no_grad():
        _, _, beam = dec_model.generate_beam(encoded, test_len, beam_size=beam_size, length_penalty=1.0, early_stopping=1, max_len=200)
        assert len(beam) == 1
    hypotheses = beam[0].hyp
    assert len(hypotheses) == beam_size


    # Convert outputs to math equations
    for score, sent in sorted(hypotheses, key=lambda x: x[0], reverse=True):

        # parse decoded hypothesis
        ids = sent[1:].tolist()                  # decoded token IDs
        tok = [env.id2word[wid] for wid in ids]  # convert to prefix

        # Currently I just remove the furst 2 tokens as the seem to break prefix_to_infix
        # TODO: change this to convert Y' to f'(x) (This should fix the breaking problem)
        if tok[:2] != ['sub', "Y'"]:
            print('invalid prefix!')

        try:
            tok_crop = tok[2:]
            hyp = env.prefix_to_infix(tok_crop)       # convert to infix
            hyp = env.infix_to_sympy(hyp)        # convert to SymPy

            res = "OK"

        except:
            res = "INVALID PREFIX EXPRESSION"
            hyp = tok

        # print result
        print("%.5f  %s  %s" % (score, res, hyp))


# Env is an overarching environment object that does a bunch of useful things
env = build_env(params)

# Parse data path
s = [x.split(',') for x in params.reload_data.split(';') if len(x) > 0]
data_path = {task: (train_path, valid_path, test_path) for task, train_path, valid_path, test_path in s}

# this both creates the dataset and the dataloader
dl = env.create_train_iterator(params.tasks[0], params, data_path)

# te = next(iter(dl))
# print(te)

# Create the transformer parts
enc_model = TransformerModel(params, env.id2word, is_encoder=True, with_output=False)
dec_model = TransformerModel(params, env.id2word, is_encoder=False, with_output=True)

# Collect all model params
named_params = []
for v in [enc_model, dec_model]:
    named_params.extend([(k, p) for k, p in v.named_parameters() if p.requires_grad])
model_params = [p for k, p in named_params]
total_params = 0
for v in model_params:
    total_params += len(v)
print("Found %i parameters." % (total_params))

optimizer = torch.optim.Adam(model_params, lr=1e-4)

# Train loop
num_epochs = 5
losses = []
for n in range(num_epochs):
    loss_accum = 0

    for i, batch in tqdm(enumerate(dl), total=250):
        
        # Train for one batch
        loss_curr = enc_dec_step(batch, optimizer, enc_model, dec_model)
        loss_accum += loss_curr.item()

        # Currently the dataloader seems to just iterate forever, not really 
        # sure whats going on with that, will need to fix

        # For now i've just said 1 'epoch' will be 250 batches
        # To break up training to allow for logging + testing
        if i >= 250:
            break
    
    # Generate some equations
    print('Epoch ', n, ' generated equation:')
    test_seq_gen(env, enc_model, dec_model)

    # Log the loss
    loss_accum /= i
    print('Loss: %.4f' % (loss_accum))
    losses.append(loss_accum)

# sys.stdout.close()

plt.plot(losses)
plt.savefig('results/loss_plot_fixedcrop.png')


