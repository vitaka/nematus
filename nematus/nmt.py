'''
Build a neural machine translation model with soft attention
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import json
import ipdb
import numpy
import copy

import os
import warnings
import sys
import time

from subprocess import Popen

from collections import OrderedDict

profile = False

from data_iterator import TextIterator
from util import *
from theano_util import *
from alignment_util import *

from layers import *
from initializers import *
from optimizers import *

from domain_interpolation_data_iterator import DomainInterpolatorTextIterator

# batch preparation
def prepare_data(seqs_x, seqs_y, maxlen=None, n_words_src=[30000],
                 n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    n_factors = len(seqs_x[0][0])
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((n_factors, maxlen_x, n_samples)).astype('int64')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:, :lengths_x[idx], idx] = zip(*s_x)
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask


# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    for factor in range(options['factors']):
        params[embedding_name(factor)] = norm_weight(options['n_words_src'][factor], options['dim_word_per_factor'][factor])

    params['Wemb_dec'] = norm_weight(options['n_words'], options['dim_word'])

    for factor in range(options['factors']):
        # encoder: bidirectional RNN
        params = get_layer_param(options['encoder'])(options, params,
                                                  prefix=factored_layer_name('encoder',factor),
                                                  nin=options['dim_word_per_factor'][factor],
                                                  dim=options['dim_per_factor'][factor])
        params = get_layer_param(options['encoder'])(options, params,
                                                  prefix=factored_layer_name('encoder_r',factor),
                                                  nin=options['dim_word_per_factor'][factor],
                                                  dim=options['dim_per_factor'][factor])

    ctxdim = 2 * options['dim']

    # init_state, init_cell
    params = get_layer_param('ff')(options, params, prefix='ff_state',
                                nin=2*sum(options['dim_per_factor']), nout=options['dim'])
    # decoder
    params = get_layer_param(options['decoder'])(options, params,
                                              prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'],
                                              dimctx=2*sum(options['dim_per_factor']))
                                              #dimctx=ctxdim)
    # readout
    params = get_layer_param('ff')(options, params, prefix='ff_logit_lstm',
                                nin=options['dim'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer_param('ff')(options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer_param('ff')(options, params, prefix='ff_logit_ctx',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False)
    params = get_layer_param('ff')(options, params, prefix='ff_logit',
                                nin=options['dim_word'],
                                nout=options['n_words'])

    return params

# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    #x has 3 dimensions because of factors
    x = tensor.tensor3('x', dtype='int64')
    x.tag.test_value = (numpy.random.rand(1, 5, 10)*100).astype('int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    x_mask.tag.test_value = numpy.ones(shape=(5, 10)).astype('float32')
    y = tensor.matrix('y', dtype='int64')
    y.tag.test_value = (numpy.random.rand(8, 10)*100).astype('int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')
    y_mask.tag.test_value = numpy.ones(shape=(8, 10)).astype('float32')

    # for the backward rnns, we just need to invert x and x_mask
    xr = x[:,::-1]
    xr_mask = x_mask[::-1]

    n_timesteps = x.shape[1]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[2]

    if options['use_dropout']:
        retain_probability_emb = 1-options['dropout_embedding']
        retain_probability_hidden = 1-options['dropout_hidden']
        retain_probability_source = 1-options['dropout_source']
        retain_probability_target = 1-options['dropout_target']
        rec_dropout_l=[]
        rec_dropout_r_l=[]
        for factor in xrange(options['factors']):
            rec_dropout_l.append(shared_dropout_layer((2, n_samples, options['dim_per_factor'][factor]), use_noise, trng, retain_probability_hidden))
            rec_dropout_r_l.append(shared_dropout_layer((2, n_samples, options['dim_per_factor'][factor]), use_noise, trng, retain_probability_hidden))
        rec_dropout_d = shared_dropout_layer((4+options['factors'], n_samples, options['dim']), use_noise, trng, retain_probability_hidden)
        emb_dropout_l = []
        for factor in xrange(options['factors']):
            emb_dropout_l.append(shared_dropout_layer((2, n_samples, options['dim_word_per_factor'][factor]), use_noise, trng, retain_probability_emb))
        emb_dropout_r_l=[]
        for factor in xrange(options['factors']):
            emb_dropout_r_l.append(shared_dropout_layer((2, n_samples, options['dim_word_per_factor'][factor]), use_noise, trng, retain_probability_emb))
        emb_dropout_d = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb)

        ctx_dropout_d_l =[]
        for factor in xrange(options['factors']):
            ctx_dropout_d_l.append(shared_dropout_layer((2, n_samples, 2*options['dim_per_factor'][factor]), use_noise, trng, retain_probability_hidden))
        ctx_dropout_d_j=shared_dropout_layer((2, n_samples, 2*sum(options['dim_per_factor'])), use_noise, trng, retain_probability_hidden)
        source_dropout_l=[]
        for factor in xrange(options['factors']):
            source_dropout_l.append(shared_dropout_layer((n_timesteps, n_samples, 1), use_noise, trng, retain_probability_source))
            source_dropout_l[-1] = tensor.tile(source_dropout_l[-1], (1,1,options['dim_word_per_factor'][factor]))
        target_dropout = shared_dropout_layer((n_timesteps_trg, n_samples, 1), use_noise, trng, retain_probability_target)
        target_dropout = tensor.tile(target_dropout, (1,1,options['dim_word']))
    else:
        rec_dropout_l=[]
        rec_dropout_r_l=[]
        emb_dropout_l=[]
        emb_dropout_r_l=[]
        for factor in xrange(options['factors']):
            rec_dropout_l.append(theano.shared(numpy.array([1.]*2, dtype='float32')))
            rec_dropout_r_l.append(theano.shared(numpy.array([1.]*2, dtype='float32')))
            emb_dropout_l.append(theano.shared(numpy.array([1.]*2, dtype='float32')))
            emb_dropout_r_l.append(theano.shared(numpy.array([1.]*2, dtype='float32')))
        rec_dropout_d = theano.shared(numpy.array([1.]*(4+options['factors']), dtype='float32'))
        emb_dropout_d = theano.shared(numpy.array([1.]*2, dtype='float32'))
        ctx_dropout_d_l = [ theano.shared(numpy.array([1.]*2, dtype='float32')) for factor in xrange(options['factors']) ]
        ctx_dropout_d_j= theano.shared(numpy.array([1.]*2, dtype='float32'))

    # word embedding for forward rnn (source)
    proj_l=[]
    for factor in range(options['factors']):
        #emb.append(tparams[embedding_name(factor)][x[factor].flatten()])
        emb=tparams[embedding_name(factor)][x[factor].flatten()]
        #emb = concatenate(emb, axis=1)
        emb = emb.reshape([n_timesteps, n_samples, options['dim_word_per_factor'][factor]])

        if options['use_dropout']:
            emb *= source_dropout_l[factor]
        proj_l.append(get_layer_constr(options['encoder'])(tparams, emb, options,
                                            prefix= factored_layer_name('encoder',factor),
                                            mask=x_mask,
                                            emb_dropout=emb_dropout_l[factor],
                                            rec_dropout=rec_dropout_l[factor],
                                            profile=profile))

    # word embedding for backward rnn (source)
    projr_l=[]
    for factor in range(options['factors']):
        #embr.append(tparams[embedding_name(factor)][xr[factor].flatten()])
        embr = tparams[embedding_name(factor)][xr[factor].flatten()]
        #embr = concatenate(embr, axis=1)
        embr = embr.reshape([n_timesteps, n_samples, options['dim_word_per_factor'][factor]])
        if options['use_dropout']:
            embr *= source_dropout_l[factor][::-1]
        projr_l.append(get_layer_constr(options['encoder'])(tparams, embr, options,
                                             prefix=factored_layer_name('encoder_r',factor),
                                             mask=xr_mask,
                                             emb_dropout=emb_dropout_r_l[factor],
                                             rec_dropout=rec_dropout_r_l[factor],
                                             profile=profile))


    #we have one context for each factor, they will be concatenated after the attention has been applied
    # context will be the concatenation of forward and backward rnns
    ctx_l = []
    for factor in xrange(options['factors']):
        ctx_l.append(concatenate([proj_l[factor][0], projr_l[factor][0][::-1]], axis=proj_l[factor][0].ndim-1))

    #TODO: think about using weights and tanh and adapt this: see section 2.1 of Barret Zoph and Kevin Knight (2016). "Multi-Source Neural Translation"
    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = concatenate([ (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None] for ctx in ctx_l ], axis=1)

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    #This droput is OK
    if options['use_dropout']:
        ctx_mean *= shared_dropout_layer((n_samples, 2*sum(options['dim_per_factor'])), use_noise, trng, retain_probability_hidden)

    # initial decoder state
    init_state = get_layer_constr('ff')(tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    emb = tparams['Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])

    emb_shifted = tensor.zeros_like(emb)
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    if options['use_dropout']:
        emb *= target_dropout

    # decoder - pass through the decoder conditional gru with attention
    proj = get_layer_constr(options['decoder'])(tparams, emb, options,
                                            prefix='decoder',
                                            mask=y_mask, context_l=ctx_l,
                                            context_mask=x_mask,
                                            one_step=False,
                                            init_state=init_state,
                                            emb_dropout=emb_dropout_d,
                                            ctx_dropout_l=ctx_dropout_d_l,
                                            ctx_dropout_j=ctx_dropout_d_j,
                                            rec_dropout=rec_dropout_d,
                                            profile=profile)
    # hidden states of the decoder gru
    proj_h = proj[0]

    # weighted averages of context, generated by attention module
    ctxs = proj[1]

    #This droput does not require any modification
    if options['use_dropout']:
        proj_h *= shared_dropout_layer((n_samples, options['dim']), use_noise, trng, retain_probability_hidden)
        emb *= shared_dropout_layer((n_samples, options['dim_word']), use_noise, trng, retain_probability_emb)
        ctxs *= shared_dropout_layer((n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden)

    # weights (alignment matrix) #####LIUCAN: this is where the attention vector is.
    # WARNING: this is now a list with one element per input factor
    opt_ret['dec_alphas'] = proj[2]

    # compute word probabilities
    logit_lstm = get_layer_constr('ff')(tparams, proj_h, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer_constr('ff')(tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer_constr('ff')(tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

    if options['use_dropout']:
        logit *= shared_dropout_layer((n_samples, options['dim_word']), use_noise, trng, retain_probability_hidden)

    logit = get_layer_constr('ff')(tparams, logit, options,
                               prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]]))

    # cost
    y_flat = y.flatten()
    y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'] + y_flat
    cost = -tensor.log(probs.flatten()[y_flat_idx])
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0)

    #print "Print out in build_model()"
    #print opt_ret
    return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost


# build a sampler
def build_sampler(tparams, options, use_noise, trng, return_alignment=False):
    x = tensor.tensor3('x', dtype='int64')
    xr = x[:,::-1]
    n_timesteps = x.shape[1]
    n_samples = x.shape[2]

    #DEBUG
    #tparams['ff_logit_b']  = numpy.zeros((options['n_words'],)).astype('float32')

    # word embedding (source), forward and backward
    emb_l = []
    embr_l = []
    for factor in range(options['factors']):
        emb_l.append(tparams[embedding_name(factor)][x[factor].flatten()])
        embr_l.append(tparams[embedding_name(factor)][xr[factor].flatten()])
    #emb = concatenate(emb, axis=1)
    #embr = concatenate(embr, axis=1)
    for factor in range(options['factors']):
        emb_l[factor] = emb_l[factor].reshape([n_timesteps, n_samples, options['dim_word_per_factor'][factor]])
        embr_l[factor] = embr_l[factor].reshape([n_timesteps, n_samples, options['dim_word_per_factor'][factor]])

    if options['use_dropout']:
        retain_probability_emb = 1-options['dropout_embedding']
        retain_probability_hidden = 1-options['dropout_hidden']
        retain_probability_source = 1-options['dropout_source']
        retain_probability_target = 1-options['dropout_target']

        rec_dropout_l=[]
        rec_dropout_r_l=[]
        for factor in range(options['factors']):
            rec_dropout_l.append(theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32')))
            rec_dropout_r_l.append(theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32')))
        rec_dropout_d = theano.shared(numpy.array([retain_probability_hidden]*(4+options['factors']), dtype='float32'))

        emb_dropout_l=[]
        emb_dropout_r_l=[]
        for factor in range(options['factors']):
            emb_dropout_l.append(theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32')))
            emb_dropout_r_l.append(theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32')))
        emb_dropout_d = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))

        ctx_dropout_d_l=[]
        for factor in xrange(options['factors']):
            ctx_dropout_d_l.append(theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32')))
        ctx_dropout_d_j=theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32'))

        source_dropout_l = [theano.shared(numpy.float32(retain_probability_source)) for factor in range(options['factors']) ]
        target_dropout = theano.shared(numpy.float32(retain_probability_target))
        for factor in range(options['factors']):
            emb_l[factor] *= source_dropout
            embr_l[factor] *= source_dropout
    else:
        rec_dropout_l = [theano.shared(numpy.array([1.]*2, dtype='float32')) for factor in range(options['factors']) ]
        rec_dropout_r_l = [ theano.shared(numpy.array([1.]*2, dtype='float32')) for factor in range(options['factors']) ]
        rec_dropout_d = theano.shared(numpy.array([1.]*(4+options['factors']), dtype='float32'))
        emb_dropout_l = [ theano.shared(numpy.array([1.]*2, dtype='float32')) for factor in range(options['factors']) ]
        emb_dropout_r_l = [ theano.shared(numpy.array([1.]*2, dtype='float32')) for factor in range(options['factors']) ]
        emb_dropout_d = theano.shared(numpy.array([1.]*2, dtype='float32'))
        ctx_dropout_d_l = [ theano.shared(numpy.array([1.]*2, dtype='float32')) for factor in range(options['factors']) ]
        ctx_dropout_d_j=theano.shared(numpy.array([1.]*2, dtype='float32'))

    # encoder
    proj_l=[]
    projr_l=[]
    for factor in range(options['factors']):
        proj_l.append(get_layer_constr(options['encoder'])(tparams, emb_l[factor], options,
                                            prefix= factored_layer_name('encoder',factor), emb_dropout=emb_dropout_l[factor], rec_dropout=rec_dropout_l[factor], profile=profile))

        projr_l.append(get_layer_constr(options['encoder'])(tparams, embr_l[factor], options,
                                             prefix= factored_layer_name('encoder_r',factor), emb_dropout=emb_dropout_r_l[factor], rec_dropout=rec_dropout_r_l[factor], profile=profile))
    ctx_l = []
    for factor in xrange(options['factors']):
    # concatenate forward and backward rnn hidden states
        ctx_l.append(concatenate([proj_l[factor][0], projr_l[factor][0][::-1]], axis=proj_l[factor][0].ndim-1))

    # get the input for decoder rnn initializer mlp
    #TODO: think about using weights and tanh and adapt this: see section 2.1 of Barret Zoph and Kevin Knight (2016). "Multi-Source Neural Translation"
    ctx_mean =  concatenate( [ ctx.mean(0) for ctx in ctx_l ], axis=1)

    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)

    if options['use_dropout']:
        ctx_mean *= retain_probability_hidden

    init_state = get_layer_constr('ff')(tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    #ctx is now a list
    print >>sys.stderr, 'Building f_init...',
    outs = [init_state ]+ctx_l
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print >>sys.stderr, 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    if options['use_dropout']:
        emb *= target_dropout

    # apply one step of conditional gru with attention
    proj = get_layer_constr(options['decoder'])(tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context_l=ctx_l,
                                            one_step=True,
                                            init_state=init_state,
                                            emb_dropout=emb_dropout_d,
                                            ctx_dropout_l=ctx_dropout_d_l,
                                            ctx_dropout_j=ctx_dropout_d_j,
                                            rec_dropout=rec_dropout_d,
                                            profile=profile)

    # get the next hidden state
    next_state = proj[0]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]

    # alignment matrix (attention model): Now it is a list
    dec_alphas = proj[2]

    #nothing new to adjust
    if options['use_dropout']:
        next_state_up = next_state * retain_probability_hidden
        emb *= retain_probability_emb
        ctxs *= retain_probability_hidden
    else:
        next_state_up = next_state

    logit_lstm = get_layer_constr('ff')(tparams, next_state_up, options,
                                    prefix='ff_logit_lstm', activ='linear')
    logit_prev = get_layer_constr('ff')(tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer_constr('ff')(tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

    if options['use_dropout']:
        logit *= retain_probability_hidden

    logit = get_layer_constr('ff')(tparams, logit, options,
                              prefix='ff_logit', activ='linear')

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print >>sys.stderr, 'Building f_next..',
    inps = [y] +ctx_l+ [init_state]
    outs = [next_probs, next_sample, next_state]

    if return_alignment:
        outs.extend(dec_alphas)

    #print >> sys.stderr, "tparams ff_logit_b: "+ str(tparams['ff_logit_b'].get_value())

    f_next = theano.function(inps, outs, name='f_next', profile=profile)
    print >>sys.stderr, 'Done'

    return f_init, f_next

# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
#TODO: return alphas
def gen_sample(f_init, f_next, x, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False, return_alignment=False, return_alphas=False, suppress_unk=False):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    sample_word_probs = []
    alignment = []
    final_alphas = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    word_probs = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []
    if return_alignment:
        hyp_alignment = [[] for _ in xrange(live_k)]
    if return_alphas:
        hyp_alphas = [[] for _ in xrange(live_k)]

    # for ensemble decoding, we keep track of states and probability distribution
    # for each model in the ensemble
    num_models = len(f_init)
    next_state = [None]*num_models
    ctx0 = [None]*num_models
    next_p = [None]*num_models
    dec_alphas = [None]*num_models
    dec_all_alphas = [None]*num_models
    # get initial state of decoder rnn and encoder context
    for i in xrange(num_models):
        ret = f_init[i](x)
        next_state[i] = ret[0]
        ctx0[i] = ret[1:]
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator

    # x is a sequence of word ids followed by 0, eos id
    for ii in xrange(maxlen):
        for i in xrange(num_models):
            ctx_l = [numpy.tile(ctx0[i][j], [live_k, 1]) for j in xrange(len(ctx0[i])) ]
            inps = [next_w] + ctx_l + [next_state[i]]
            ret = f_next[i](*inps)
            # dimension of dec_alpha (k-beam-size, number-of-input-hidden-units)
            next_p[i], next_w_tmp, next_state[i] = ret[0], ret[1], ret[2]
            if return_alignment:
                #these are the alphas for the first factor
                dec_alphas[i] = ret[3]
            if return_alphas:
                dec_all_alphas[i] = ret[3:]

            if suppress_unk:
                next_p[i][:,1] = -numpy.inf
        if stochastic:
            if argmax:
                nw = sum(next_p)[0].argmax()
            else:
                nw = next_w_tmp[0]
            sample.append(nw)
            sample_score += numpy.log(next_p[0][0, nw])
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - sum(numpy.log(next_p))
            probs = sum(next_p)/num_models
            cand_flat = cand_scores.flatten()
            probs_flat = probs.flatten()
            ranks_flat = cand_flat.argpartition(k-dead_k-1)[:(k-dead_k)]

            #averaging the attention weights across models
            #TODO: we only consider the first factor to compute alignment (value returned by f_next) Maybe an average would be more convenient?
            if return_alignment:
                mean_alignment = sum([dec_alphas[i] for i in xrange(num_models) ])/num_models

            if return_alphas:
                num_factors=len(dec_all_alphas[0])
                alphas_for_this_word=[]
                for factor in xrange(num_factors):
                    alphas_for_this_word.append(sum([ dec_all_alphas[i][factor] for i in xrange(num_models) ])/num_models)

            voc_size = next_p[0].shape[1]
            # index of each k-best hypothesis
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_word_probs = []
            new_hyp_states = []
            if return_alignment:
                # holds the history of attention weights for each time step for each of the surviving hypothesis
                # dimensions (live_k * target_words * source_hidden_units]
                # at each time step we append the attention weights corresponding to the current target word
                new_hyp_alignment = [[] for _ in xrange(k-dead_k)]

            if return_alphas:
                new_hyp_alphas = [[] for _ in xrange(k-dead_k)]

            # ti -> index of k-best hypothesis
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_word_probs.append(word_probs[ti] + [probs_flat[ranks_flat[idx]].tolist()])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append([copy.copy(next_state[i][ti]) for i in xrange(num_models)])
                if return_alignment:
                    # get history of attention weights for the current hypothesis
                    new_hyp_alignment[idx] = copy.copy(hyp_alignment[ti])
                    # extend the history with current attention weights
                    new_hyp_alignment[idx].append(mean_alignment[ti])
                if return_alphas:
                    # get history of attention weights for the current hypothesis TODO: ??
                    new_hyp_alphas[idx] = copy.copy(hyp_alphas[ti])
                    # extend the history with current attention weights TODO: ??
                    new_hyp_alphas[idx].append([alphas_for_this_word[i][ti] for i in xrange(len(alphas_for_this_word)) ])


            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            word_probs = []
            if return_alignment:
                hyp_alignment = []
            if return_alphas:
                hyp_alphas = []

            # sample and sample_score hold the k-best translations and their scores
            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    sample_word_probs.append(new_word_probs[idx])
                    if return_alignment:
                        alignment.append(new_hyp_alignment[idx])
                    if return_alphas:
                        final_alphas.append(new_hyp_alphas[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    word_probs.append(new_word_probs[idx])
                    if return_alignment:
                        hyp_alignment.append(new_hyp_alignment[idx])
                    if return_alphas:
                        hyp_alphas.append(new_hyp_alphas[idx])
            hyp_scores = numpy.array(hyp_scores)

            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = [numpy.array(state) for state in zip(*hyp_states)]

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
                sample_word_probs.append(word_probs[idx])
                if return_alignment:
                    alignment.append(hyp_alignment[idx])
                if return_alphas:
                    final_alphas.append(hyp_alphas[idx])

    if not return_alignment:
        alignment = [None for i in range(len(sample))]
    if not return_alphas:
        final_alphas=[None for i in range(len(sample))]

    return sample, sample_score, sample_word_probs, alignment, final_alphas


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True, normalize=False, alignweights=False):
    probs = []
    n_done = 0

    alignments_json = []

    for x, y in iterator:
        #ensure consistency in number of factors
        if len(x[0][0]) != options['factors']:
            sys.stderr.write('Error: mismatch between number of factors in settings ({0}), and number in validation corpus ({1})\n'.format(options['factors'], len(x[0][0])))
            sys.exit(1)

        n_done += len(x)

        x, x_mask, y, y_mask = prepare_data(x, y,
                                            n_words_src=options['n_words_src'],
                                            n_words=options['n_words'])

        ### in optional save weights mode.
        if alignweights:
            pprobs, attention = f_log_probs(x, x_mask, y, y_mask)
            for jdata in get_alignments(attention, x_mask, y_mask):
                alignments_json.append(jdata)
        else:
            pprobs = f_log_probs(x, x_mask, y, y_mask)

        # normalize scores according to output length
        if normalize:
            lengths = numpy.array([numpy.count_nonzero(s) for s in y_mask.T])
            pprobs /= lengths

        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs), alignments_json


def train(dim_word=100,  # word vector dimensionality
          dim=1000,  # the number of LSTM units
          dim_per_factor=None,
          factors=1, # input factors
          dim_word_per_factor=None, # list of word vector dimensionalities (one per factor): [250,200,50] for total dimensionality of 500
          encoder='gru',
          decoder='gru_cond',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          map_decay_c=0., # L2 regularization penalty towards original weights
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=0.01,  # learning rate
          n_words_src=None,  # source vocabulary size
          n_words=None,  # target vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok'],
          valid_datasets=['../data/dev/newstest2011.en.tok',
                          '../data/dev/newstest2011.fr.tok'],
          dictionaries=[
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.en.tok.pkl',
              '/data/lisatmp3/chokyun/europarl/europarl-v7.fr-en.fr.tok.pkl'],
          use_dropout=False,
          dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
          dropout_hidden=0.5, # dropout for hidden layers (0: no dropout)
          dropout_source=0, # dropout source words (0: no dropout)
          dropout_target=0, # dropout target words (0: no dropout)
          reload_=False,
          overwrite=False,
          external_validation_script=None,
          shuffle_each_epoch=True,
          finetune=False,
          finetune_only_last=False,
          sort_by_length=True,
          use_domain_interpolation=False,
          domain_interpolation_min=0.1,
          domain_interpolation_inc=0.1,
          domain_interpolation_indomain_datasets=['indomain.en', 'indomain.fr'],
          maxibatch_size=20): #How many minibatches to load at one time

    # Model options
    model_options = locals().copy()


    if model_options['dim_word_per_factor'] == None:
        if factors == 1:
            model_options['dim_word_per_factor'] = [model_options['dim_word']]
        else:
            sys.stderr.write('Error: if using factored input, you must specify \'dim_word_per_factor\'\n')
            sys.exit(1)

    if model_options['dim_per_factor'] == None:
        if factors == 1:
            model_options['dim_per_factor'] = [model_options['dim']]
        else:
            sys.stderr.write('Error: if using factored input, you must specify \'dim_per_factor\'\n')
            sys.exit(1)


    assert(len(dictionaries) == factors + 1) # one dictionary per source factor + 1 for target factor
    assert(len(model_options['dim_word_per_factor']) == factors) # each factor embedding has its own dimensionality
    assert(sum(model_options['dim_word_per_factor']) == model_options['dim_word']) # dimensionality of factor embeddings sums up to total dimensionality of input embedding vector
    assert(factors <= MAXFACTORS) #we must not exceed maximum number of supported factors

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        worddicts[ii] = load_dict(dd)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    if n_words_src is None:
        n_words_src=[]
        for i in xrange(len(factors)):
            n_words_src.append(len(worddicts[i]))
        model_options['n_words_src'] = n_words_src
    if n_words is None:
	n_words = len(worddicts[1])
        model_options['n_words'] = n_words

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'Reloading model options'
        try:
            with open('%s.json' % saveto, 'rb') as f:
                loaded_model_options = json.load(f)
                model_options.update(loaded_model_options)
        except:
            pass

    print 'Loading data'
    domain_interpolation_cur = None
    if use_domain_interpolation:
        print 'Using domain interpolation with initial ratio %s, increase rate %s' % (domain_interpolation_min, domain_interpolation_inc)
        domain_interpolation_cur = domain_interpolation_min
        train = DomainInterpolatorTextIterator(datasets[0], datasets[1],
                         dictionaries[:-1], dictionaries[1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen,
                         shuffle_each_epoch=shuffle_each_epoch,
                         sort_by_length=sort_by_length,
                         indomain_source=domain_interpolation_indomain_datasets[0],
                         indomain_target=domain_interpolation_indomain_datasets[1],
                         interpolation_rate=domain_interpolation_cur,
                         maxibatch_size=maxibatch_size)
    else:
        train = TextIterator(datasets[0], datasets[1],
                         dictionaries[:-1], dictionaries[-1],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen,
                         shuffle_each_epoch=shuffle_each_epoch,
                         sort_by_length=sort_by_length,
                         maxibatch_size=maxibatch_size)

    if valid_datasets and validFreq:
        valid = TextIterator(valid_datasets[0], valid_datasets[1],
                            dictionaries[:-1], dictionaries[-1],
                            n_words_source=n_words_src, n_words_target=n_words,
                            batch_size=valid_batch_size,
                            maxlen=maxlen)
    else:
        valid = None

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        print 'Reloading model parameters'
        params = load_params(saveto, params)

    tparams = init_theano_params(params)

    #print >> sys.stderr, "ff_logit_b: "+ str(params['ff_logit_b'])
    #print >> sys.stderr, "tparams ff_logit_b: "+ str(tparams['ff_logit_b'].get_value())

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)

    inps = [x, x_mask, y, y_mask]

    #print >> sys.stderr, "ff_logit_b: "+ str(params['ff_logit_b'])
    #print >> sys.stderr, "tparams ff_logit_b: "+ str(tparams['ff_logit_b'].get_value())

    if validFreq or sampleFreq:
        print 'Building sampler'
        f_init, f_next = build_sampler(tparams, model_options, use_noise, trng)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

     # apply L2 regularisation to loaded model (map training)
    if map_decay_c > 0:
        map_decay_c = theano.shared(numpy.float32(map_decay_c), name="map_decay_c")
        weight_map_decay = 0.
        for kk, vv in tparams.iteritems():
            init_value = theano.shared(vv.get_value(), name= kk + "_init")
            weight_map_decay += ((vv -init_value) ** 2).sum()
        weight_map_decay *= map_decay_c
        cost += weight_map_decay

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    # allow finetuning with fixed embeddings
    if finetune:
        updated_params = OrderedDict([(key,value) for (key,value) in tparams.iteritems() if not key.startswith('Wemb')])
    else:
        updated_params = tparams

    # allow finetuning of only last layer (becomes a linear model training problem)
    if finetune_only_last:
        updated_params = OrderedDict([(key,value) for (key,value) in tparams.iteritems() if key in ['ff_logit_W', 'ff_logit_b']])
    else:
        updated_params = tparams

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(updated_params))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')

    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, updated_params, grads, inps, cost, profile=profile)
    print 'Done'

    print 'Optimization'

    best_p = None
    bad_counter = 0
    uidx = 0
    estop = False
    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        rmodel = numpy.load(saveto)
        history_errs = list(rmodel['history_errs'])
        if 'uidx' in rmodel:
            uidx = rmodel['uidx']

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    if sampleFreq == -1:
        sampleFreq = len(train[0])/batch_size

    valid_err = None

    for eidx in xrange(max_epochs):
        n_samples = 0

        for x, y in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            #ensure consistency in number of factors
            if len(x) and len(x[0]) and len(x[0][0]) != factors:
                sys.stderr.write('Error: mismatch between number of factors in settings ({0}), and number in training corpus ({1})\n'.format(factors, len(x[0][0])))
                sys.exit(1)

            x, x_mask, y, y_mask = prepare_data(x, y, maxlen=maxlen,
                                                n_words_src=n_words_src,
                                                n_words=n_words)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask, y, y_mask)

            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            # save the best model so far, in addition, save the latest model
            # into a separate file with the iteration number for external eval
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving the best model...',
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip_from_theano(tparams)
                numpy.savez(saveto, history_errs=history_errs, uidx=uidx, **params)
                json.dump(model_options, open('%s.json' % saveto, 'wb'), indent=2)
                print 'Done'

                # save with uidx
                if not overwrite:
                    print 'Saving the model at iteration {}...'.format(uidx),
                    saveto_uidx = '{}.iter{}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)
                    numpy.savez(saveto_uidx, history_errs=history_errs,
                                uidx=uidx, **unzip_from_theano(tparams))
                    print 'Done'


            # generate some samples with the model and display them
            if sampleFreq and numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                for jj in xrange(numpy.minimum(5, x.shape[2])):
                    stochastic = True
                    sample, score, sample_word_probs, alignment = gen_sample([f_init], [f_next],
                                               x[:, :, jj][:, :, None],
                                               trng=trng, k=1,
                                               maxlen=30,
                                               stochastic=stochastic,
                                               argmax=False,
                                               suppress_unk=False)
                    print 'Source ', jj, ': ',
                    for pos in range(x.shape[1]):
                        if x[0, pos, jj] == 0:
                            break
                        for factor in range(factors):
                            vv = x[factor, pos, jj]
                            if vv in worddicts_r[factor]:
                                sys.stdout.write(worddicts_r[factor][vv])
                            else:
                                sys.stdout.write('UNK')
                            if factor+1 < factors:
                                sys.stdout.write('|')
                            else:
                                sys.stdout.write(' ')
                    print
                    print 'Truth ', jj, ' : ',
                    for vv in y[:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[-1]:
                            print worddicts_r[-1][vv],
                        else:
                            print 'UNK',
                    print
                    print 'Sample ', jj, ': ',
                    if stochastic:
                        ss = sample
                    else:
                        score = score / numpy.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                    for vv in ss:
                        if vv == 0:
                            break
                        if vv in worddicts_r[-1]:
                            print worddicts_r[-1][vv],
                        else:
                            print 'UNK',
                    print

            # validate model on validation set and early stop if necessary
            if valid and validFreq and numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs, alignment = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)

                if uidx == 0 or valid_err <= numpy.array(history_errs).min():
                    best_p = unzip_from_theano(tparams)
                    bad_counter = 0
                if len(history_errs) > patience and valid_err >= \
                        numpy.array(history_errs)[:-patience].min():
                    bad_counter += 1
                    if bad_counter > patience:
                        if use_domain_interpolation and (domain_interpolation_cur < 1.0):
                            domain_interpolation_cur = min(domain_interpolation_cur + domain_interpolation_inc, 1.0)
                            print 'No progress on the validation set, increasing domain interpolation rate to %s and resuming from best params' % domain_interpolation_cur
                            train.adjust_domain_interpolation_rate(domain_interpolation_cur)
                            if best_p is not None:
                                zip_to_theano(best_p, tparams)
                            bad_counter = 0
                        else:
                            print 'Early Stop!'
                            estop = True
                            break

                if numpy.isnan(valid_err):
                    ipdb.set_trace()

                print 'Valid ', valid_err

                if external_validation_script:
                    print "Calling external validation script"
                    print 'Saving  model...',
                    params = unzip_from_theano(tparams)
                    numpy.savez(saveto +'.dev', history_errs=history_errs, uidx=uidx, **params)
                    json.dump(model_options, open('%s.dev.npz.json' % saveto, 'wb'), indent=2)
                    print 'Done'
                    p = Popen([external_validation_script])

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        #This is the end of an epoch. Save model
        if not overwrite:
            print 'Saving the model at the end of epoch  {}...'.format(eidx),
            saveto_uidx = '{}.epoch{}.npz'.format(
                os.path.splitext(saveto)[0], eidx)
            numpy.savez(saveto_uidx, history_errs=history_errs,
                        uidx=uidx, **unzip_from_theano(tparams))
            print 'Done'

        if estop:
            break

    if best_p is not None:
        zip_to_theano(best_p, tparams)

    if valid:
        use_noise.set_value(0.)
        valid_errs, alignment = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
        valid_err = valid_errs.mean()

        print 'Valid ', valid_err

    if best_p is not None:
        params = copy.copy(best_p)
    else:
        params = unzip_from_theano(tparams)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                uidx=uidx,
                **params)

    return valid_err


if __name__ == '__main__':
    pass
