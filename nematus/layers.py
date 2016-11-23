'''
Layer definitions
'''

import sys

import json
import cPickle as pkl
import numpy
from collections import OrderedDict

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from initializers import *
from util import *
from theano_util import *
from alignment_util import *

MAXFACTORS=6

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          #'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          'gru_cond': ('param_init_gru_cond_multiple_encoders', 'gru_cond_layer_multiple_encoders'),
          }

def get_layer_param(name):
    param_fn, constr_fn = layers[name]
    return eval(param_fn)

def get_layer_constr(name):
    param_fn, constr_fn = layers[name]
    return eval(constr_fn)

# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj

# dropout that will be re-used at different time steps
def shared_dropout_layer(shape, use_noise, trng, value):
    proj = tensor.switch(
        use_noise,
        trng.binomial(shape, p=value, n=1,
                                     dtype='float32'),
        theano.shared(numpy.float32(value)))
    return proj

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[pp(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[pp(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[pp(prefix, 'W')]) +
        tparams[pp(prefix, 'b')])


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[pp(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[pp(prefix, 'Wx')] = Wx
    params[pp(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[pp(prefix, 'Ux')] = Ux

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None,
              emb_dropout=None,
              rec_dropout=None,
              profile=False,
              **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[pp(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below*emb_dropout[0], tparams[pp(prefix, 'W')]) + \
        tparams[pp(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below*emb_dropout[1], tparams[pp(prefix, 'Wx')]) + \
        tparams[pp(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux, rec_dropout):

        preact = tensor.dot(h_*rec_dropout[0], U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_*rec_dropout[1], Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]
    _step = _step_slice
    shared_vars = [tparams[pp(prefix, 'U')],
                   tparams[pp(prefix, 'Ux')],
                   rec_dropout]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=pp(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[pp(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[pp(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux')] = Ux
    params[pp(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype('float32')

    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[pp(prefix, 'U_nl')] = U_nl
    params[pp(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype('float32')

    Ux_nl = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux_nl')] = Ux_nl
    params[pp(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype('float32')

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2)
    params[pp(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[pp(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[pp(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[pp(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[pp(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[pp(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[pp(prefix, 'c_tt')] = c_att

    return params

# Conditional GRU layer with Attention
#TODO: define parameters that are independent for earch encoder
def param_init_gru_cond_multiple_encoders(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[pp(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[pp(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux')] = Ux
    params[pp(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype('float32')

    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[pp(prefix, 'U_nl')] = U_nl
    params[pp(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype('float32')

    Ux_nl = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux_nl')] = Ux_nl
    params[pp(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype('float32')

    # context to LSTM
    Wc = norm_weight(dimctx, dim*2)
    params[pp(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[pp(prefix, 'Wcx')] = Wcx

    #IMPORTANT: In the paper by Cho, the same size for n and n' is used. Here, it seems that in the default implementation
    #in this piece of software n' = 2n

    #different for each encoder
    # attention: combined -> hidden
    for factor in xrange(options['factors']):
        #W_comb_att = norm_weight(dim, dimctx)
        W_comb_att = norm_weight(options['dim'], options['dim_per_factor'][factor]*2)
        params[factored_layer_name(pp(prefix, 'W_comb_att'),factor)] = W_comb_att

    #different for each encoder
    # attention: context -> hidden
    for factor in xrange(options['factors']):
        #Wc_att = norm_weight(dimctx)
        Wc_att = norm_weight(options['dim_per_factor'][factor]*2)
        params[factored_layer_name(pp(prefix, 'Wc_att'),factor)] = Wc_att

    #different for each encoder
    # attention: hidden bias
    for factor in xrange(options['factors']):
        #b_att = numpy.zeros((dimctx,)).astype('float32')
        b_att = numpy.zeros((options['dim_per_factor'][factor]*2,)).astype('float32')
        params[factored_layer_name(pp(prefix, 'b_att'),factor)] = b_att

    #different for each encoder
    # attention:
    for factor in xrange(options['factors']):
        #U_att = norm_weight(dimctx, 1)
        U_att = norm_weight(options['dim_per_factor'][factor]*2, 1)
        params[factored_layer_name(pp(prefix, 'U_att'),factor)] = U_att

        c_att = numpy.zeros((1,)).astype('float32')
        params[factored_layer_name(pp(prefix, 'c_tt'),factor)] = c_att

    return params


def gru_cond_layer(tparams, state_below, options, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None, emb_dropout=None,
                   rec_dropout=None, ctx_dropout=None,
                   profile=False,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    #state_below is E dot y_i (for all i? Yes, with theano.scan we loop over them)
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[pp(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3-d: #annotation x #sample x dim'
    #pctx_ = vector with U_a dot h_j for all j
    pctx_ = tensor.dot(context*ctx_dropout[0], tparams[pp(prefix, 'Wc_att')]) +\
        tparams[pp(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    #W_z dot E dot y_i  , W_r dot E dot y_i. One value for each TL position
    state_belowx = tensor.dot(state_below*emb_dropout[0], tparams[pp(prefix, 'Wx')]) +\
        tparams[pp(prefix, 'bx')]
    state_below_ = tensor.dot(state_below*emb_dropout[1], tparams[pp(prefix, 'W')]) +\
        tparams[pp(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, pctx_, cc_, rec_dropout, ctx_dropout,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):

        # WARNING: this is slightly different from the equations shown in the paper NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
        # here there are kind of 2 steps, one without context and the other one with context

        #WE DON'T NEED TO CHANGE ANYTHING IN THE CALCULATION OF h1, BECAUSE CONTEXT IS NOT INVOLVED
        #it seems that this is the previous timestep or something, because r and u are computed again later
        #z_i and r_i in paper. Computed at the same time
        preact1 = tensor.dot(h_*rec_dropout[0], U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        #r_i
        r1 = _slice(preact1, 0, dim)

        #z_i
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_*rec_dropout[1], Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        #pctx_ = vector with U_a dot h_j for all j. It is computed outside this function
        #because it does not depend on the current decoder state
        # attention

        #WE NEED TO COMPUTE ONE ctx_ FOR EACH ENCODER, AND THEN CONCATENATE THEM

        #W_a dot s_{i-1}
        pstate_ = tensor.dot(h1*rec_dropout[2], W_comb_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        #U_att = v_a^T
        alpha = tensor.dot(pctx__*ctx_dropout[1], U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))
        if context_mask:
            alpha = alpha * context_mask
        #alpha for all j (SL sentence positions)
        alpha = alpha / alpha.sum(0, keepdims=True)

        #c_i in paper: cc_ = h_j for all j?
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        #CONCATENATE ALL THE ctx_

        #b_nl should be W_zr dot E dot y_i, but it seems that it is a bias. What is going on?
        #They have been already added when computig preact1
        preact2 = tensor.dot(h1*rec_dropout[3], U_nl)+b_nl
        preact2 += tensor.dot(ctx_*ctx_dropout[2], Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1*rec_dropout[4], Ux_nl)+bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_*ctx_dropout[3], Wcx)

        h2 = tensor.tanh(preactx2)

        #the line below corresponds to the first equation in section A.2.2
        h2 = u2 * h1 + (1. - u2) * h2

        #and this one? It is the mask
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[pp(prefix, 'U')],
                   tparams[pp(prefix, 'Wc')],
                   tparams[pp(prefix, 'W_comb_att')],
                   tparams[pp(prefix, 'U_att')],
                   tparams[pp(prefix, 'c_tt')],
                   tparams[pp(prefix, 'Ux')],
                   tparams[pp(prefix, 'Wcx')],
                   tparams[pp(prefix, 'U_nl')],
                   tparams[pp(prefix, 'Ux_nl')],
                   tparams[pp(prefix, 'b_nl')],
                   tparams[pp(prefix, 'bx_nl')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context, rec_dropout, ctx_dropout] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[0])],
                                    non_sequences=[pctx_, context, rec_dropout, ctx_dropout]+shared_vars,
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


def gru_cond_layer_multiple_encoders(tparams, state_below, options, prefix='gru',
                   mask=None, context_l=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None, emb_dropout=None,
                   rec_dropout=None, ctx_dropout_l=None,ctx_dropout_j=None,
                   profile=False,
                   **kwargs):

    assert context_l, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    #state_below is E dot y_i (for all i? Yes, with theano.scan we loop over them)
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    #do we need different Wcx??? No
    dim = tparams[pp(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert all(context.ndim == 3 for context in context_l ), \
        'Context must be 3-d: #annotation x #sample x dim'

    #pctx_ = vector with U_a dot h_j for all j
    pctx_l =  [ tensor.dot(context*ctx_dropout_l[factor][0], tparams[ factored_layer_name(pp(prefix, 'Wc_att'),factor) ] ) + tparams[ factored_layer_name(pp(prefix, 'b_att'),factor) ] for factor,context in enumerate(context_l) ]


    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    #W_z dot E dot y_i  , W_r dot E dot y_i. One value for each TL position. below_ for first pass, below_x for second pass
    state_belowx = tensor.dot(state_below*emb_dropout[0], tparams[pp(prefix, 'Wx')]) +\
        tparams[pp(prefix, 'bx')]
    state_below_ = tensor.dot(state_below*emb_dropout[1], tparams[pp(prefix, 'W')]) +\
        tparams[pp(prefix, 'b')]

    #TODO: solution: create a version of scan that works with the maxmum number of allowed factors

    def _step_slice_0(m_, x_, xx_, h_, ctx_, alpha_first_factor, pctx_0, cc_0,  rec_dropout, ctx_dropout_0,ctx_dropout_j,
                    U, Wc, W_comb_att_0, U_att_0, c_tt_0, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):
        cc_l=[cc_0]
        pctx_l=[pctx_0]
        ctx_dropout_l=[ctx_dropout_0]
        W_comb_att_l=[W_comb_att_0]
        U_att_l=[U_att_0]
        c_tt_l=[c_tt_0]

        return _step_slice(m_, x_, xx_, h_, ctx_, alpha_first_factor, pctx_l, cc_l, rec_dropout, ctx_dropout_l,ctx_dropout_j,
                        U, Wc, W_comb_att_l, U_att_l, c_tt_l, Ux, Wcx,
                        U_nl, Ux_nl, b_nl, bx_nl)

    def _step_slice_1(m_, x_, xx_, h_, ctx_, alpha_first_factor, pctx_0, pctx_1, cc_0, cc_1, rec_dropout, ctx_dropout_0,ctx_dropout_1,ctx_dropout_j,
                    U, Wc, W_comb_att_0,W_comb_att_1, U_att_0,U_att_1, c_tt_0,c_tt_1, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):
        cc_l=[cc_0,cc_1]
        pctx_l=[pctx_0, pctx_1]
        ctx_dropout_l=[ctx_dropout_0,ctx_dropout_1]
        W_comb_att_l=[W_comb_att_0,W_comb_att_1]
        U_att_l=[U_att_0,U_att_1]
        c_tt_l=[c_tt_0,c_tt_1]

        return _step_slice(m_, x_, xx_, h_, ctx_, alpha_first_factor, pctx_l, cc_l, rec_dropout, ctx_dropout_l,ctx_dropout_j,
                        U, Wc, W_comb_att_l, U_att_l, c_tt_l, Ux, Wcx,
                        U_nl, Ux_nl, b_nl, bx_nl)

    def _step_slice_2(m_, x_, xx_, h_, ctx_, alpha_first_factor, pctx_0, pctx_1, pctx_2, cc_0, cc_1, cc_2, rec_dropout, ctx_dropout_0,ctx_dropout_1, ctx_dropout_2, ctx_dropout_j,
                    U, Wc, W_comb_att_0,W_comb_att_1, W_comb_att_2, U_att_0,U_att_1, U_att_2, c_tt_0,c_tt_1, c_tt_2, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):
        cc_l=[cc_0,cc_1,cc_2]
        pctx_l=[pctx_0, pctx_1 , pctx_2 ]
        ctx_dropout_l=[ctx_dropout_0,ctx_dropout_1 ,ctx_dropout_2]
        W_comb_att_l=[W_comb_att_0,W_comb_att_1, W_comb_att_2]
        U_att_l=[U_att_0,U_att_1,U_att_2]
        c_tt_l=[c_tt_0,c_tt_1,c_tt_2]

        return _step_slice(m_, x_, xx_, h_, ctx_, alpha_first_factor, pctx_l, cc_l, rec_dropout, ctx_dropout_l,ctx_dropout_j,
                        U, Wc, W_comb_att_l, U_att_l, c_tt_l, Ux, Wcx,
                        U_nl, Ux_nl, b_nl, bx_nl)

    def _step_slice_3(m_, x_, xx_, h_, ctx_, alpha_first_factor, pctx_0, pctx_1, pctx_2, pctx_3,  cc_0, cc_1, cc_2, cc_3, rec_dropout, ctx_dropout_0,ctx_dropout_1, ctx_dropout_2, ctx_dropout_3, ctx_dropout_j,
                    U, Wc, W_comb_att_0,W_comb_att_1, W_comb_att_2, W_comb_att_3, U_att_0,U_att_1, U_att_2, U_att_3, c_tt_0,c_tt_1, c_tt_2, c_tt_3, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):
        cc_l=[cc_0,cc_1,cc_2,cc_3]
        pctx_l=[pctx_0, pctx_1 , pctx_2 , pctx_3 ]
        ctx_dropout_l=[ctx_dropout_0,ctx_dropout_1 ,ctx_dropout_2 ,ctx_dropout_3]
        W_comb_att_l=[W_comb_att_0,W_comb_att_1, W_comb_att_2 , W_comb_att_3]
        U_att_l=[U_att_0,U_att_1,U_att_2,U_att_3]
        c_tt_l=[c_tt_0,c_tt_1,c_tt_2,c_tt_3]

        return _step_slice(m_, x_, xx_, h_, ctx_, alpha_first_factor, pctx_l, cc_l, rec_dropout, ctx_dropout_l,ctx_dropout_j,
                        U, Wc, W_comb_att_l, U_att_l, c_tt_l, Ux, Wcx,
                        U_nl, Ux_nl, b_nl, bx_nl)

    def _step_slice_4(m_, x_, xx_, h_, ctx_, alpha_first_factor, pctx_0, pctx_1, pctx_2, pctx_3, pctx_4,  cc_0, cc_1, cc_2, cc_3, cc_4, rec_dropout, ctx_dropout_0,ctx_dropout_1, ctx_dropout_2, ctx_dropout_3, ctx_dropout_4, ctx_dropout_j, U, Wc, W_comb_att_0,W_comb_att_1, W_comb_att_2, W_comb_att_3, W_comb_att_4, U_att_0,U_att_1, U_att_2, U_att_3, U_att_4, c_tt_0,c_tt_1, c_tt_2, c_tt_3, c_tt_4, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):
        cc_l=[cc_0,cc_1,cc_2,cc_3,cc_4]
        pctx_l=[pctx_0, pctx_1 , pctx_2 , pctx_3, pctx_4 ]
        ctx_dropout_l=[ctx_dropout_0,ctx_dropout_1 ,ctx_dropout_2 ,ctx_dropout_3,ctx_dropout_4]
        W_comb_att_l=[W_comb_att_0,W_comb_att_1, W_comb_att_2 , W_comb_att_3, W_comb_att_4]
        U_att_l=[U_att_0,U_att_1,U_att_2,U_att_3,U_att_4]
        c_tt_l=[c_tt_0,c_tt_1,c_tt_2,c_tt_3,c_tt_4]

        return _step_slice(m_, x_, xx_, h_, ctx_, alpha_first_factor, pctx_l, cc_l, rec_dropout, ctx_dropout_l,ctx_dropout_j,
                        U, Wc, W_comb_att_l, U_att_l, c_tt_l, Ux, Wcx,
                        U_nl, Ux_nl, b_nl, bx_nl)

    def _step_slice_5(m_, x_, xx_, h_, ctx_, alpha_first_factor, pctx_0, pctx_1, pctx_2, pctx_3, pctx_4, pctx_5,  cc_0, cc_1, cc_2, cc_3, cc_4, cc_5, rec_dropout, ctx_dropout_0,ctx_dropout_1, ctx_dropout_2, ctx_dropout_3, ctx_dropout_4, ctx_dropout_5, ctx_dropout_j, U, Wc, W_comb_att_0,W_comb_att_1, W_comb_att_2, W_comb_att_3, W_comb_att_4, W_comb_att_5, U_att_0,U_att_1, U_att_2, U_att_3, U_att_4, U_att_5, c_tt_0,c_tt_1, c_tt_2, c_tt_3, c_tt_4, c_tt_5, Ux, Wcx,U_nl, Ux_nl, b_nl, bx_nl):
        cc_l=[cc_0,cc_1,cc_2,cc_3,cc_4,cc_5]
        pctx_l=[pctx_0, pctx_1 , pctx_2 , pctx_3, pctx_4 , pctx_5 ]
        ctx_dropout_l=[ctx_dropout_0,ctx_dropout_1 ,ctx_dropout_2 ,ctx_dropout_3,ctx_dropout_4,ctx_dropout_5]
        W_comb_att_l=[W_comb_att_0,W_comb_att_1, W_comb_att_2 , W_comb_att_3, W_comb_att_4, W_comb_att_5]
        U_att_l=[U_att_0,U_att_1,U_att_2,U_att_3,U_att_4,U_att_5]
        c_tt_l=[c_tt_0,c_tt_1,c_tt_2,c_tt_3,c_tt_4,c_tt_5]

        return _step_slice(m_, x_, xx_, h_, ctx_, alpha_first_factor, pctx_l, cc_l, rec_dropout, ctx_dropout_l,ctx_dropout_j,
                        U, Wc, W_comb_att_l, U_att_l, c_tt_l, Ux, Wcx,
                        U_nl, Ux_nl, b_nl, bx_nl)

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_first_factor, pctx_l, cc_l, rec_dropout, ctx_dropout_l,ctx_dropout_j,
                    U, Wc, W_comb_att_l, U_att_l, c_tt_l, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl):

        # WARNING: this is slightly different from the equations shown in the paper NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
        # here there are kind of 2 steps, one without context and the other one with context

        #WE DON'T NEED TO CHANGE ANYTHING IN THE CALCULATION OF h1, BECAUSE CONTEXT IS NOT INVOLVED
        #it seems that this is the previous timestep or something, because r and u are computed again later
        #z_i and r_i in paper. Computed at the same time
        preact1 = tensor.dot(h_*rec_dropout[0], U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        #r_i
        r1 = _slice(preact1, 0, dim)

        #z_i
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_*rec_dropout[1], Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        #pctx_ = vector with U_a dot h_j for all j. It is computed outside this function
        #because it does not depend on the current decoder state
        # attention

        #WE NEED TO COMPUTE ONE ctx_ FOR EACH ENCODER, AND THEN CONCATENATE THEM
        #rec_dropout has been adjusted
        ctx_l=[]
        alpha_l=[]

        #print >>sys.stderr, "Inside _step_slice"
        #print >>sys.stderr, str(pctx_l)
        #for i,t in enumerate(pctx_l):
        #    print >> sys.stderr, str(i)+" "+str(t)

        for factor,pctx_ in enumerate(pctx_l):
            #W_a dot s_{i-1}
            pstate_ = tensor.dot(h1*rec_dropout[2+factor], W_comb_att_l[factor])
            pctx__ = pctx_ + pstate_[None, :, :] #extend pstate_ to all the SL contexts
            #pctx__ += xc_
            pctx__ = tensor.tanh(pctx__)
            #U_att = v_a^T
            alpha = tensor.dot(pctx__ * ctx_dropout_l[factor][1], U_att_l[factor])+c_tt_l[factor]
            alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
            alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))
            #TODO: now there is a single context mask. We assume that the length of all the input sequences is the
            #same one. This shuld be changed if we want to have inputs with different lengths
            if context_mask:
                alpha = alpha * context_mask
            #alpha for all j (SL sentence positions)
            alpha = alpha / alpha.sum(0, keepdims=True)
            alpha_l.append(alpha)

            #c_i in paper: cc_ = h_j for all j?
            ctx_l.append ( (cc_l[factor] * alpha[:, :, None]).sum(0) ) # current context

        #CONCATENATE ALL THE ctx_l to build ctx_
        #Each member of ctx_l has dimensionality #samples x #2 * dim (a different dim for each of them)
        ctx_=concatenate(ctx_l,axis=1)

        # issues with ctx_dropout. IN some places we need a different dropout for each factor.
        #in other places, a singke one. dimensionality also changes
        #
        #new approach ctx_dropout_l: one element in the list for each factor. Each element contains 2 values
        #ctx_dropout_j: 2 values

        #b_nl should be W_zr dot E dot y_i, but it seems that it is a bias. What is going on?
        #They have been already added when computig preact1
        preact2 = tensor.dot(h1*rec_dropout[3+len(pctx_l)-1], U_nl)+b_nl
        preact2 += tensor.dot(ctx_*ctx_dropout_j[0], Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1*rec_dropout[4+len(pctx_l)-1], Ux_nl)+bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_*ctx_dropout_j[1], Wcx)

        h2 = tensor.tanh(preactx2)

        #the line below corresponds to the first equation in section A.2.2
        h2 = u2 * h1 + (1. - u2) * h2

        #and this one? It is the mask
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1


        return h2, ctx_, [alpha_l[i].T for in in xrange(len(pctx_l)) ]  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]

    functionDict={ 1: _step_slice_0, 2: _step_slice_1, 3: _step_slice_2 , 4: _step_slice_3 , 5: _step_slice_4 , 6: _step_slice_5  }

    _step = functionDict[len(context_l)]

    shared_vars = [tparams[pp(prefix, 'U')],
                   tparams[pp(prefix, 'Wc')] ] + [ tparams[factored_layer_name(pp(prefix, 'W_comb_att'),factor)] for factor,context in enumerate(context_l) ] + [tparams[factored_layer_name(pp(prefix, 'U_att'),factor)] for factor,context in enumerate(context_l) ] + [ tparams[factored_layer_name(pp(prefix, 'c_tt'),factor)] for factor,context in enumerate(context_l) ] + [tparams[pp(prefix, 'Ux')],
                   tparams[pp(prefix, 'Wcx')],
                   tparams[pp(prefix, 'U_nl')],
                   tparams[pp(prefix, 'Ux_nl')],
                   tparams[pp(prefix, 'b_nl')],
                   tparams[pp(prefix, 'bx_nl')]]



    if False:
        print >>sys.stderr, str(pctx_l)
        for i,t in enumerate(pctx_l):
            print >> sys.stderr, str(i)+" "+str(t)

    if one_step:
        rval = _step(*(seqs + [init_state, None, None ]  + pctx_l + context_l +  [ rec_dropout] + ctx_dropout_l + [ctx_dropout_j] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               dim*2),#prev ctx_ context.shape[2] = length of context dim
                                                 tensor.alloc(0., n_samples,
                                                               context_l[0].shape[0])], #prev alpha. context.shape[0] = maxlen
                                    non_sequences=  pctx_l + context_l + [rec_dropout] + ctx_dropout_l  + [ctx_dropout_j]+shared_vars,
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval
