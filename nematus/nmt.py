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

import theano.sandbox.cuda.basic_ops as sbcuda

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
def prepare_data(seqs_x, seqs_y, maxlen=None, n_words_src=30000,
                 n_words=[30000]):
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
    n_factors_tl=len(seqs_y[0][0])
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((n_factors, maxlen_x, n_samples)).astype('int64')
    y_l = [numpy.zeros((maxlen_y, n_samples)).astype('int64') for i in xrange(n_factors_tl) ]
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask_l =  [ numpy.zeros((maxlen_y, n_samples)).astype('float32') for i in xrange(n_factors_tl) ]
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:, :lengths_x[idx], idx] = zip(*s_x)
        x_mask[:lengths_x[idx]+1, idx] = 1.

        for factor in xrange(n_factors_tl):
            y_l[factor][:lengths_y[idx], idx] = [ w[factor] for w in s_y ]
            y_mask_l[factor][:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y_l, y_mask_l


# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    for factor in range(options['factors']):
        params[embedding_name(factor)] = norm_weight(options['n_words_src'], options['dim_per_factor'][factor])

    for factor in range(options['factors_tl']):
        if options['do_not_train_surface'] == True and factor == options['factors_tl']-1:
            break
        params[embedding_name(factor)+'_dec'] = norm_weight(options['n_words'][factor], options['dim_word_per_factor_tl'][factor])


    # encoder: bidirectional RNN. Shared across TL factors
    params = get_layer_param(options['encoder'])(options, params,
                                              prefix='encoder',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    params = get_layer_param(options['encoder'])(options, params,
                                              prefix='encoder_r',
                                              nin=options['dim_word'],
                                              dim=options['dim'])
    ctxdim = 2 * options['dim']

    for factor in xrange(options['factors_tl']):
        if options['do_not_train_surface'] == True and factor == options['factors_tl']-1:
            break
        # init_state, init_cell
        params = get_layer_param('ff')(options, params, prefix=factored_layer_name('ff_state',factor),
                                    nin=ctxdim, nout=options['dim_per_factor_tl'][factor])

        if factor == options['factors_tl'] -1:
            params = get_layer_param('gru_2inputs')(options, params,
                                                      prefix='rnngenerator',
                                                      nin=options['dim_word_per_factor_tl'][factor],
                                                      ninc=sum(options['dim_word_per_factor_tl'][:-1]),
                                                      dim=options['dim_per_factor_tl'][factor])

        else:
            # decoder
            params = get_layer_param(options['decoder'])(options, params,
                                                      prefix=factored_layer_name('decoder',factor),
                                                      nin=options['dim_word_per_factor_tl'][factor],
                                                      dim=options['dim_per_factor_tl'][factor],
                                                      dimctx=ctxdim)
        # readout
        params = get_layer_param('ff')(options, params, prefix=factored_layer_name('ff_logit_lstm',factor),
                                    nin=options['dim_per_factor_tl'][factor], nout=options['dim_word_per_factor_tl'][factor],
                                    ortho=False)
        params = get_layer_param('ff')(options, params, prefix=factored_layer_name('ff_logit_prev',factor),
                                    nin=options['dim_word_per_factor_tl'][factor],
                                    nout=options['dim_word_per_factor_tl'][factor], ortho=False)

        if factor == options['factors_tl'] -1:
            params = get_layer_param('ff')(options, params,prefix=factored_layer_name('ff_logit_ctx',factor),
                                    nin=sum(options['dim_word_per_factor_tl'][:-1]), nout=options['dim_word_per_factor_tl'][factor],
                                    ortho=False)
        else:
            params = get_layer_param('ff')(options, params,prefix=factored_layer_name('ff_logit_ctx',factor),
                                    nin=ctxdim, nout=options['dim_word_per_factor_tl'][factor],
                                    ortho=False)
        params = get_layer_param('ff')(options, params, prefix=factored_layer_name('ff_logit',factor),
                                    nin=options['dim_word_per_factor_tl'][factor],
                                    nout=options['n_words'][factor])

    #final FF that generates surface forms
    #params = get_layer_param('ff')(options, params, prefix='ff_generator',
    #                            nin=sum(options['dim_word_per_factor_tl'][:-1]),
    #                            nout=options['n_words'][-1]) #last factor has not associated decoder and depends on the other factors

    return params


# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.tensor3('x', dtype='int64')
    x.tag.test_value = (numpy.random.rand(1, 5, 10)*100).astype('int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')
    x_mask.tag.test_value = numpy.ones(shape=(5, 10)).astype('float32')

    # for the backward rnn, we just need to invert x and x_mask
    xr = x[:,::-1]
    xr_mask = x_mask[::-1]

    n_timesteps = x.shape[1]
    n_samples = x.shape[2]

    if options['use_dropout']:
        retain_probability_emb = 1-options['dropout_embedding']
        retain_probability_hidden = 1-options['dropout_hidden']
        retain_probability_source = 1-options['dropout_source']
        retain_probability_target = 1-options['dropout_target']
        rec_dropout = shared_dropout_layer((2, n_samples, options['dim']), use_noise, trng, retain_probability_hidden)
        rec_dropout_r = shared_dropout_layer((2, n_samples, options['dim']), use_noise, trng, retain_probability_hidden)
        rec_dropout_d_l=[]
        for factor in xrange(options['factors_tl']):
            rec_dropout_d_l.append(shared_dropout_layer((5, n_samples, options['dim_per_factor_tl'][factor]), use_noise, trng, retain_probability_hidden))
        emb_dropout = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb)
        emb_dropout_r = shared_dropout_layer((2, n_samples, options['dim_word']), use_noise, trng, retain_probability_emb)
        emb_dropout_d_l=[]
        for factor in xrange(options['factors_tl']):
            emb_dropout_d_l.append(shared_dropout_layer((2, n_samples, options['dim_word_per_factor_tl'][factor]), use_noise, trng, retain_probability_emb))
        emb_dropout_d_factors=shared_dropout_layer((2, n_samples, sum(options['dim_word_per_factor_tl'][:-1])), use_noise, trng, retain_probability_emb)

        ctx_dropout_d_l= []
        for factor in xrange(options['factors_tl']-1):
            ctx_dropout_d.append(shared_dropout_layer((4, n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden))
        source_dropout = shared_dropout_layer((n_timesteps, n_samples, 1), use_noise, trng, retain_probability_source)
        source_dropout = tensor.tile(source_dropout, (1,1,options['dim_word']))

    else:
        rec_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
        rec_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
        rec_dropout_d_l = [ theano.shared(numpy.array([1.]*5, dtype='float32')) for i in xrange(options['factors_tl']) ]
        emb_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
        emb_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
        emb_dropout_d_l = [ theano.shared(numpy.array([1.]*2, dtype='float32')) for i in xrange(options['factors_tl']) ]
        emb_dropout_d_factors= theano.shared(numpy.array([1.]*2, dtype='float32'))
        ctx_dropout_d_l = [ theano.shared(numpy.array([1.]*4, dtype='float32')) for i in xrange(options['factors_tl']-1) ]

    # word embedding for forward rnn (source)
    emb = []
    for factor in range(options['factors']):
        emb.append(tparams[embedding_name(factor)][x[factor].flatten()])
    emb = concatenate(emb, axis=1)
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    if options['use_dropout']:
        emb *= source_dropout

    proj = get_layer_constr(options['encoder'])(tparams, emb, options,
                                            prefix='encoder',
                                            mask=x_mask,
                                            emb_dropout=emb_dropout,
                                            rec_dropout=rec_dropout,
                                            profile=profile)


    # word embedding for backward rnn (source)
    embr = []
    for factor in range(options['factors']):
        embr.append(tparams[embedding_name(factor)][xr[factor].flatten()])
    embr = concatenate(embr, axis=1)
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])
    if options['use_dropout']:
        embr *= source_dropout[::-1]

    projr = get_layer_constr(options['encoder'])(tparams, embr, options,
                                             prefix='encoder_r',
                                             mask=xr_mask,
                                             emb_dropout=emb_dropout_r,
                                             rec_dropout=rec_dropout_r,
                                             profile=profile)

    # context will be the concatenation of forward and backward rnns
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    if options['use_dropout']:
        ctx_mean *= shared_dropout_layer((n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden)


    y_l=[]
    y_mask_l=[]
    cost_l=[]
    probs_l=[]

    #shape of y: n_timesteps x n_samples
    #word_embeddings_tl = n_words x dim_word

    for factor in xrange(options['factors_tl']-1):
        y = tensor.matrix( factored_layer_name('y',factor) , dtype='int64')
        y.tag.test_value = (numpy.random.rand(8, 10)*100).astype('int64')
        y_mask = tensor.matrix( factored_layer_name('y_mask',factor) , dtype='float32')
        y_mask.tag.test_value = numpy.ones(shape=(8, 10)).astype('float32')

        n_timesteps_trg = y.shape[0]

        if options['use_dropout']:
            target_dropout=shared_dropout_layer((n_timesteps_trg, n_samples, 1), use_noise, trng, retain_probability_target)
            target_dropout = tensor.tile(target_dropout, (1,1,options['dim_word_per_factor_tl'][factor]))

        # initial decoder state
        init_state = get_layer_constr('ff')(tparams, ctx_mean, options,
                                    prefix=factored_layer_name('ff_state',factor), activ='tanh')

        # word embedding (target), we will shift the target sequence one time step
        # to the right. This is done because of the bi-gram connections in the
        # readout and decoder rnn. The first target will be all zeros and we will
        # not condition on the last output.
        emb = tparams[ embedding_name(factor)+'_dec'][y.flatten()]
        emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word_per_factor_tl'][factor]])

        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted

        if options['use_dropout']:
            emb *= target_dropout

        # decoder - pass through the decoder conditional gru with attention
        proj = get_layer_constr(options['decoder'])(tparams, emb, options,
                                            prefix=factored_layer_name('decoder',factor),
                                            mask=y_mask, context=ctx,
                                            context_mask=x_mask,
                                            one_step=False,
                                            init_state=init_state,
                                            emb_dropout=emb_dropout_d_l[factor],
                                            ctx_dropout=ctx_dropout_d_l[factor],
                                            rec_dropout=rec_dropout_d_l[factor],
                                            profile=profile)
        # hidden states of the decoder gru
        proj_h = proj[0]

        # weighted averages of context, generated by attention module
        ctxs = proj[1]

        if options['use_dropout']:
            proj_h *= shared_dropout_layer((n_samples, options['dim_per_factor_tl'][factor]), use_noise, trng, retain_probability_hidden)
            emb *= shared_dropout_layer((n_samples,options['dim_word_per_factor_tl'][factor]), use_noise, trng, retain_probability_emb)
            ctxs *= shared_dropout_layer((n_samples, 2*options['dim']), use_noise, trng, retain_probability_hidden)

        # weights (alignment matrix) #####LIUCAN: this is where the attention vector is.
        opt_ret[ factored_layer_name('dec_alphas',factor) ] = proj[2]

        # compute word probabilities
        logit_lstm = get_layer_constr('ff')(tparams, proj_h, options,
                                        prefix=factored_layer_name('ff_logit_lstm',factor), activ='linear')
        logit_prev = get_layer_constr('ff')(tparams, emb, options,
                                        prefix=factored_layer_name('ff_logit_prev',factor), activ='linear')
        logit_ctx = get_layer_constr('ff')(tparams, ctxs, options,
                                       prefix=factored_layer_name('ff_logit_ctx',factor), activ='linear')
        logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

        if options['use_dropout']:
            logit *= shared_dropout_layer((n_samples, options['dim_word_per_factor_tl'][factor]), use_noise, trng, retain_probability_hidden)

        logit = get_layer_constr('ff')(tparams, logit, options,
                                   prefix=factored_layer_name('ff_logit',factor), activ='linear')
        #From Theano help: The softmax function will, when applied to a matrix, compute the softmax values row-wise.
        logit_shp = logit.shape
        #logit_shp: n_timesteps x n_samples x n_words
        #probs: n_timesteps*n_samples x n_words
        probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                                   logit_shp[2]]))
        #n_words is different for each factor
        # cost
        y_flat = y.flatten()
        y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'][factor] + y_flat
        cost = -tensor.log(probs.flatten()[y_flat_idx])
        cost = cost.reshape([y.shape[0], y.shape[1]])
        cost = (cost * y_mask).sum(0)

        y_l.append(y)
        y_mask_l.append(y_mask)
        cost_l.append(cost)
        probs_l.append(probs)


    if options['do_not_train_surface'] == False:
        #for each factor, make product of probs and embeddings, concatenate them all
        #and feed the generator feed-forward
        generator_input_l=[]
        for factor in xrange(options['factors_tl']-1):
            #emb: n_timesteps * n_samples x dim_word_per_factor_tl[factor]
            emb = tensor.dot(probs_l[factor],tparams[ embedding_name(factor)+'_dec']).reshape([n_timesteps_trg, n_samples,options['dim_word_per_factor_tl'][factor]])
            generator_input_l.append(emb)
        generator_input=tensor.concatenate(generator_input_l,axis=-1)


    #last factor: surface form
    factor=options['factors_tl']-1
    #Our generator is a GRU-based RNN: at each timestep, its
    #inputs are the embeddings of the previously generated surface form,
    #and the embeddings of all the factors generated by the decoders.

    y = tensor.matrix( factored_layer_name('y',factor) , dtype='int64')
    y.tag.test_value = (numpy.random.rand(8, 10)*100).astype('int64')
    y_mask = tensor.matrix( factored_layer_name('y_mask',factor) , dtype='float32')
    y_mask.tag.test_value = numpy.ones(shape=(8, 10)).astype('float32')
    y_l.append(y)
    y_mask_l.append(y_mask)


    if options['do_not_train_surface'] == False:
        emb = tparams[ embedding_name(factor)+'_dec'][y.flatten()]
        emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word_per_factor_tl'][factor]])

        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted


        if options['use_dropout']:
            target_dropout=shared_dropout_layer((n_timesteps_trg, n_samples, 1), use_noise, trng, retain_probability_target)
            target_dropout = tensor.tile(target_dropout, (1,1,options['dim_word_per_factor_tl'][factor]))
            emb *= target_dropout

        # initial decoder state
        init_state = get_layer_constr('ff')(tparams, ctx_mean, options,
                                    prefix=factored_layer_name('ff_state',factor), activ='tanh')

        #surface form decoder
        final_proj = get_layer_constr('gru_2inputs')(tparams, emb, generator_input, options,
                                            prefix='rnngenerator',
                                            mask=y_mask,
                                            one_step=False,
                                            init_state=init_state,
                                            emb_dropout=emb_dropout_d_l[factor],
                                            emb_factor_dropout=emb_dropout_d_factors,
                                            rec_dropout=rec_dropout_d_l[factor],
                                            profile=profile)

        proj_h=final_proj[0]
        if options['use_dropout']:
            proj_h *= shared_dropout_layer((n_samples, options['dim_per_factor_tl'][factor]), use_noise, trng, retain_probability_hidden)
            emb *= shared_dropout_layer((n_samples,options['dim_word_per_factor_tl'][factor]), use_noise, trng, retain_probability_emb)
            generator_input *= shared_dropout_layer((n_samples, sum(options['dim_word_per_factor_tl'][:-1])), use_noise, trng, retain_probability_hidden)

        # compute word probabilities
        logit_lstm = get_layer_constr('ff')(tparams, proj_h, options,
                                        prefix=factored_layer_name('ff_logit_lstm',factor), activ='linear')
        logit_prev = get_layer_constr('ff')(tparams, emb, options,
                                        prefix=factored_layer_name('ff_logit_prev',factor), activ='linear')
        logit_ctx = get_layer_constr('ff')(tparams, generator_input, options,
                                       prefix=factored_layer_name('ff_logit_ctx',factor), activ='linear')
        logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

        if options['use_dropout']:
            logit *= shared_dropout_layer((n_samples, options['dim_word_per_factor_tl'][factor]), use_noise, trng, retain_probability_hidden)

        logit = get_layer_constr('ff')(tparams, logit, options,
                                   prefix=factored_layer_name('ff_logit',factor), activ='linear')
        #From Theano help: The softmax function will, when applied to a matrix, compute the softmax values row-wise.
        logit_shp = logit.shape
        #logit_shp: n_timesteps x n_samples x n_words
        #probs: n_timesteps*n_samples x n_words
        generator_probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                                   logit_shp[2]]))

        #compute cost of last factor; compute final cost
        #shape of y: n_timesteps x n_samples

        y_flat = y.flatten()
        y_flat_idx = tensor.arange(y_flat.shape[0]) * options['n_words'][factor] + y_flat
        cost = -tensor.log(generator_probs.flatten()[y_flat_idx])
        cost = cost.reshape([y.shape[0], y.shape[1]])
        cost = (cost * y_mask).sum(0)
    else:
        cost=None

    #cost per sample
    final_cost = options['lambda_parameter']*sum(cost_l)/len(cost_l)
    if options['do_not_train_surface'] == False:
        final_cost = final_cost + cost

    #print "Print out in build_model()"
    #print opt_ret

    #return trng, use_noise, x, x_mask, y, y_mask, opt_ret, cost
    return trng, use_noise, x, x_mask, y_l, y_mask_l, opt_ret, final_cost, cost, cost_l


# build a sampler
def build_sampler(tparams, options, use_noise, trng, return_alignment=False):
    x = tensor.tensor3('x', dtype='int64')
    xr = x[:,::-1]
    n_timesteps = x.shape[1]
    n_samples = x.shape[2]

    # word embedding (source), forward and backward
    emb = []
    embr = []
    for factor in range(options['factors']):
        emb.append(tparams[embedding_name(factor)][x[factor].flatten()])
        embr.append(tparams[embedding_name(factor)][xr[factor].flatten()])
    emb = concatenate(emb, axis=1)
    embr = concatenate(embr, axis=1)
    emb = emb.reshape([n_timesteps, n_samples, options['dim_word']])
    embr = embr.reshape([n_timesteps, n_samples, options['dim_word']])

    if options['use_dropout']:
        retain_probability_emb = 1-options['dropout_embedding']
        retain_probability_hidden = 1-options['dropout_hidden']
        retain_probability_source = 1-options['dropout_source']
        retain_probability_target = 1-options['dropout_target']
        rec_dropout = theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32'))
        rec_dropout_r = theano.shared(numpy.array([retain_probability_hidden]*2, dtype='float32'))
        rec_dropout_d_l = [theano.shared(numpy.array([retain_probability_hidden]*5, dtype='float32')) for i in xrange(options['factors_tl'])]
        emb_dropout = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
        emb_dropout_r = theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))
        emb_dropout_d_l = [ theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32')) for i in xrange(options['factors_tl']) ]
        emb_dropout_d_factors=theano.shared(numpy.array([retain_probability_emb]*2, dtype='float32'))

        ctx_dropout_d_l = [ theano.shared(numpy.array([retain_probability_hidden]*4, dtype='float32')) for i in xrange(options['factors_tl']-1) ]
        emb_dropout_d_factors=shared_dropout_layer((2, n_samples, sum(options['dim_word_per_factor_tl'][:-1])), use_noise, trng, retain_probability_emb)
        source_dropout = theano.shared(numpy.float32(retain_probability_source))
        target_dropout_l = [theano.shared(numpy.float32(retain_probability_target)) for i in xrange(options['factors_tl']) ]
        emb *= source_dropout
        embr *= source_dropout
    else:
        rec_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
        rec_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
        rec_dropout_d_l = [theano.shared(numpy.array([1.]*5, dtype='float32')) for i in xrange(options['factors_tl'])]
        emb_dropout = theano.shared(numpy.array([1.]*2, dtype='float32'))
        emb_dropout_r = theano.shared(numpy.array([1.]*2, dtype='float32'))
        emb_dropout_d_l = [theano.shared(numpy.array([1.]*2, dtype='float32')) for i in xrange(options['factors_tl'])]
        emb_dropout_d_factors=theano.shared(numpy.array([1.]*2, dtype='float32'))
        ctx_dropout_d_l = [theano.shared(numpy.array([1.]*4, dtype='float32')) for i in xrange(options['factors_tl']-1)]

    # encoder
    proj = get_layer_constr(options['encoder'])(tparams, emb, options,
                                            prefix='encoder', emb_dropout=emb_dropout, rec_dropout=rec_dropout, profile=profile)


    projr = get_layer_constr(options['encoder'])(tparams, embr, options,
                                             prefix='encoder_r', emb_dropout=emb_dropout_r, rec_dropout=rec_dropout_r, profile=profile)

    # concatenate forward and backward rnn hidden states
    ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)

    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(0)
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)

    if options['use_dropout']:
        ctx_mean *= retain_probability_hidden

    init_state_l=[]

    init_state_inputs_l=[]
    y_l=[]

    next_state_l=[]
    next_probs_l=[]

    dec_alphas_l=[]

    for factor in xrange(options['factors_tl']-1):
        init_state = get_layer_constr('ff')(tparams, ctx_mean, options,
                                    prefix=factored_layer_name('ff_state',factor), activ='tanh')
        init_state_l.append(init_state)

        # x: 1 x 1
        y = tensor.vector(factored_layer_name('y_sampler',factor), dtype='int64')
        init_state = tensor.matrix(factored_layer_name('init_state',factor), dtype='float32')

        y_l.append(y)
        init_state_inputs_l.append(init_state)

        # if it's the first word, emb should be all zero and it is indicated by -1
        emb = tensor.switch(y[:, None] < 0,
                            tensor.alloc(0., 1, tparams[embedding_name(factor)+'_dec'].shape[1]),
                            tparams[embedding_name(factor)+'_dec'][y])

        if options['use_dropout']:
            emb *= target_dropout_l[factor]

        # apply one step of conditional gru with attention
        proj = get_layer_constr(options['decoder'])(tparams, emb, options,
                                                prefix=factored_layer_name('decoder',factor),
                                                mask=None, context=ctx,
                                                one_step=True,
                                                init_state=init_state,
                                                emb_dropout=emb_dropout_d_l[factor],
                                                ctx_dropout=ctx_dropout_d_l[factor],
                                                rec_dropout=rec_dropout_d_l[factor],
                                                profile=profile)
        # get the next hidden state
        next_state = proj[0]
        next_state_l.append(next_state)

        # get the weighted averages of context for this target word y
        ctxs = proj[1]

        # alignment matrix (attention model)
        dec_alphas = proj[2]
        dec_alphas_l.append(dec_alphas)

        if options['use_dropout']:
            next_state_up = next_state * retain_probability_hidden
            emb *= retain_probability_emb
            ctxs *= retain_probability_hidden
        else:
            next_state_up = next_state

        logit_lstm = get_layer_constr('ff')(tparams, next_state_up, options,
                                        prefix=factored_layer_name('ff_logit_lstm',factor), activ='linear')
        logit_prev = get_layer_constr('ff')(tparams, emb, options,
                                        prefix=factored_layer_name('ff_logit_prev',factor), activ='linear')
        logit_ctx = get_layer_constr('ff')(tparams, ctxs, options,
                                       prefix=factored_layer_name('ff_logit_ctx',factor), activ='linear')
        logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

        if options['use_dropout']:
            logit *= retain_probability_hidden

        logit = get_layer_constr('ff')(tparams, logit, options,
                              prefix=factored_layer_name('ff_logit',factor), activ='linear')

        # compute the softmax probability
        next_probs = tensor.nnet.softmax(logit)
        next_probs_l.append(next_probs)


    init_state = get_layer_constr('ff')(tparams, ctx_mean, options,
                                prefix=factored_layer_name('ff_state',options['factors_tl']-1), activ='tanh')
    init_state_l.append(init_state)

    print >>sys.stderr, 'Building f_init'
    outs = init_state_l + [ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile)
    print >>sys.stderr, 'Done'

    # x: 1 x 1
    y = tensor.vector(factored_layer_name('y_sampler',options['factors_tl']-1), dtype='int64')
    init_state = tensor.matrix(factored_layer_name('init_state',options['factors_tl']-1), dtype='float32')
    y_l.append(y)
    init_state_inputs_l.append(init_state)


    #surface form generator
    generator_input_l=[]
    for factor in xrange(options['factors_tl']-1):
        #emb: n_timesteps * n_samples x dim_word_per_factor_tl[factor]
        emb = tensor.dot(next_probs_l[factor],tparams[ embedding_name(factor)+'_dec'])
        generator_input_l.append(emb)
    generator_input=tensor.concatenate(generator_input_l,axis=-1)

    factor=options['factors_tl']-1

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams[embedding_name(factor)+'_dec'].shape[1]),
                        tparams[embedding_name(factor)+'_dec'][y])

    if options['use_dropout']:
        emb *= target_dropout_l[factor]

    # apply one step of surface form decoder
    final_proj = get_layer_constr('gru_2inputs')(tparams, emb, generator_input, options,
                                        prefix='rnngenerator',
                                        mask=None,
                                        one_step=True,
                                        init_state=init_state,
                                        emb_dropout=emb_dropout_d_l[factor],
                                        emb_factor_dropout=emb_dropout_d_factors,
                                        rec_dropout=rec_dropout_d_l[factor],
                                        profile=profile)

    next_state = final_proj[0]
    next_state_l.append(next_state)

    if options['use_dropout']:
        next_state_up = next_state * retain_probability_hidden
        emb *= retain_probability_emb
    else:
        next_state_up = next_state

    logit_lstm = get_layer_constr('ff')(tparams, next_state_up, options,
                                    prefix=factored_layer_name('ff_logit_lstm',factor), activ='linear')
    logit_prev = get_layer_constr('ff')(tparams, emb, options,
                                    prefix=factored_layer_name('ff_logit_prev',factor), activ='linear')
    logit_ctx = get_layer_constr('ff')(tparams, generator_input, options,
                                   prefix=factored_layer_name('ff_logit_ctx',factor), activ='linear')
    logit = tensor.tanh(logit_lstm+logit_prev+logit_ctx)

    if options['use_dropout']:
        logit *= retain_probability_hidden

    logit = get_layer_constr('ff')(tparams, logit, options,
                          prefix=factored_layer_name('ff_logit',factor), activ='linear')

    #generator output has 1 dimenstion here, but code works anyway
    generator_probs= tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=generator_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print >>sys.stderr, 'Building f_next'
    inps = y_l + [ctx] + init_state_inputs_l
    outs = [generator_probs, next_sample] + next_state_l + next_probs_l
    #TODO: y_l has as many items as factors
    #init_state_inputs_l as as many items as factors
    #next_state_l has as many items as factors

    if return_alignment:
        outs.extend(dec_alphas_l)

    f_next = theano.function(inps, outs, name=factored_layer_name('f_next',factor), profile=profile)
    print >>sys.stderr, 'Done'

    #return f_init, f_next
    return f_init, f_next

# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(f_init, f_next, x, factors_tl=1, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False, return_alignment=False, suppress_unk=False, inversegenerationdict=dict()):
    DEBUG=False

    if DEBUG:
        print >> sys.stderr, "{} entries in dict".format(len(inversegenerationdict))
        #print >> sys.stderr, ";".join( [str(k) for k in inversegenerationdict.keys()]  )
        print >> sys.stderr, "11 is in dict: "+str(11 in inversegenerationdict)
        print >> sys.stderr, "11 str is in dict: "+str("11" in inversegenerationdict)


    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    sample_word_probs = []
    alignment = []
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

    # for ensemble decoding, we keep track of states and probability distribution
    # for each model in the ensemble
    num_models = len(f_init)
    next_state_l = [None]*num_models
    ctx0 = [None]*num_models
    next_p = [None]*num_models
    next_p_factors=[None]*num_models
    dec_alphas = [None]*num_models
    # get initial state of decoder rnn and encoder context
    for i in xrange(num_models):
        ret = f_init[i](x)
        #next_state_l[i] is a list with one state per factor
        next_state_l[i] = ret[:-1]
        ctx0[i] = ret[-1]
    next_w_l = [ -1 * numpy.ones((1,)).astype('int64') for factor in xrange(factors_tl) ]  # bos indicator

    # x is a sequence of word ids followed by 0, eos id
    for ii in xrange(maxlen):
        for i in xrange(num_models):
            ctx = numpy.tile(ctx0[i], [live_k, 1])
            inps = next_w_l + [ctx ] + next_state_l[i]
            #outs = [next_probs, next_sample] + next_state_l
            if DEBUG:
                print >>sys.stderr, "next_w_l: {} next_state_l[i]: {}".format(len(next_w_l),len(next_state_l[i]))
                print >>sys.stderr, "Calling f_next[{}] with {} elements".format(i,len(inps))
            ret = f_next[i](*inps)
            # dimension of dec_alpha (k-beam-size, number-of-input-hidden-units)
            next_p[i], next_w_tmp, next_state_l[i], next_p_factors[i] = ret[0], ret[1], ret[2:2+factors_tl],ret[2+factors_tl:2+factors_tl+factors_tl-1]
            if DEBUG:
                print >> sys.stderr, "Output position {}.Distribution of prob. for next word, hypothesis 0".format(ii)
                for myi in xrange(len(next_p[i][0])):
                    print >> sys.stderr, str(myi)+":"+str(next_p[i][0][myi])

            if return_alignment:
                #we only consider alphas of first factor
                dec_alphas[i] = ret[2+factors_tl+factors_tl-1]
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
            #hyp_scores is a vector of live_k scores
            #hyp_scores[:, None] contains live_k rows and 1 column
            #next_p[i]= live_k x n_words
            cand_scores = hyp_scores[:, None] - sum(numpy.log(next_p))
            probs = sum(next_p)/num_models
            cand_flat = cand_scores.flatten()
            probs_flat = probs.flatten()
            ranks_flat = cand_flat.argpartition(k-dead_k-1)[:(k-dead_k)]
            #one element per TL factor: each element is nparray: live_k x n_words
            probs_factors=[]
            for factor in xrange(factors_tl-1):
                probs_factors.append( sum(next_p_factors[i][factor] for i in xrange(num_models))/num_models)
            #probs_factors=  [ sum(next_p_factors[:][factor])/num_models  for factor in xrange(factors_tl-1) ]

            #averaging the attention weights accross models
            if return_alignment:
                mean_alignment = sum(dec_alphas)/num_models

            voc_size = next_p[0].shape[1]
            # index of each k-best hypothesis
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_word_probs = []
            #one element per hypothesis
            new_word_probs_factors = []
            new_hyp_states = []
            if return_alignment:
                # holds the history of attention weights for each time step for each of the surviving hypothesis
                # dimensions (live_k * target_words * source_hidden_units]
                # at each time step we append the attention weights corresponding to the current target word
                new_hyp_alignment = [[] for _ in xrange(k-dead_k)]

            # ti -> index of k-best hypothesis
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_word_probs.append(word_probs[ti] + [probs_flat[ranks_flat[idx]].tolist()])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_word_probs_factors.append([ probs_factors[factor][ti]  for factor in xrange(factors_tl-1) ])

                #new_hyp_states.append([copy.copy(next_state_l[i][ti]) for i in xrange(num_models)])
                new_element=[]
                for i in xrange(num_models):
                    factorlist=[]
                    for factor in xrange(factors_tl):
                        factorlist.append(copy.copy(next_state_l[i][factor][ti]))
                    new_element.append(factorlist)
                new_hyp_states.append(new_element)


                if return_alignment:
                    # get history of attention weights for the current hypothesis
                    new_hyp_alignment[idx] = copy.copy(hyp_alignment[ti])
                    # extend the history with current attention weights
                    new_hyp_alignment[idx].append(mean_alignment[ti])

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            word_probs = []
            word_probs_factors =[]
            if return_alignment:
                hyp_alignment = []

            # sample and sample_score hold the k-best translations and their scores
            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    sample_word_probs.append(new_word_probs[idx])
                    if return_alignment:
                        alignment.append(new_hyp_alignment[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    word_probs.append(new_word_probs[idx])
                    word_probs_factors.append(new_word_probs_factors[idx])
                    if return_alignment:
                        hyp_alignment.append(new_hyp_alignment[idx])
            hyp_scores = numpy.array(hyp_scores)

            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            # we are using a dictionary to select which intermediate
            # factors have generated the final surface forms and
            # use them as an input to the decoders.
            # for homograph words, we choose the combination of factors
            # with the highest probability
            #
            #next_w = numpy.array([w[-1] for w in hyp_samples])
            next_w_l=[ [] for factor in xrange(factors_tl) ]
            #first inefficient version:
            for hyp_idx,w in enumerate(hyp_samples):
                # we will always find the word in the dictionary since
                # we are operating with numbers and OOVs are already mapped to UNK
                analyses=inversegenerationdict[str(w[-1])]
                if len(analyses) == 1:
                    for factor in xrange(factors_tl-1):
                        next_w_l[factor].append(analyses[0][factor])
                else:
                    #the surface form we have generated is ambiguous:
                    #find the most likely analysis in the prob distribution
                    #of the factors
                    analysis_probs=[]
                    for analysis in analyses:
                        analysis_probs.append(   sum(  word_probs_factors[hyp_idx][factor][analysis[factor]] for factor in xrange(factors_tl-1))   )
                    max_idx=numpy.argmax(numpy.array(analysis_probs))
                    for factor in xrange(factors_tl-1):
                        next_w_l[factor].append(int(analyses[max_idx][factor]))
                next_w_l[factors_tl-1].append(w[-1])
            for i in xrange(len(next_w_l)):
                next_w_l[i]=numpy.array(next_w_l[i])

            next_state_l = []
            for state in zip(*hyp_states):
                resultForThisModel=[]
                for factor in xrange(factors_tl):
                    hypsForFactor=numpy.array( [hypstate[factor] for hypstate in state ] )
                    resultForThisModel.append(hypsForFactor)
                next_state_l.append(resultForThisModel)
            #[ [numpy.array(factor_state) for factor_state in state ] for state in zip(*hyp_states) ]

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
                sample_word_probs.append(word_probs[idx])
                if return_alignment:
                    alignment.append(hyp_alignment[idx])

    if not return_alignment:
        alignment = [None for i in range(len(sample))]

    return sample, sample_score, sample_word_probs, alignment


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

        x, x_mask, y_l, y_mask_l = prepare_data(x, y,
                                            n_words_src=options['n_words_src'],
                                            n_words=options['n_words'])

        ### in optional save weights mode.
        if alignweights:
            #TODO: this part does not work
            pprobs, attention = f_log_probs(*([x, x_mask]+ y_l + y_mask_l))
            for jdata in get_alignments(attention, x_mask, y_mask):
                alignments_json.append(jdata)
        else:
            pprobs = f_log_probs(*([x, x_mask]+ y_l + y_mask_l))

        # normalize scores according to output length
        if normalize:
            lengths = numpy.array([numpy.count_nonzero(s) for s in y_mask[0].T])
            pprobs /= lengths

        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs), alignments_json

# calculate the log probablities on a given corpus using translation model
def pred_probs_index(f_log_probs, prepare_data, options, iterator, index, verbose=True, normalize=False, alignweights=False):
    probs = []
    n_done = 0

    alignments_json = []

    for x, y in iterator:
        #ensure consistency in number of factors
        if len(x[0][0]) != options['factors']:
            sys.stderr.write('Error: mismatch between number of factors in settings ({0}), and number in validation corpus ({1})\n'.format(options['factors'], len(x[0][0])))
            sys.exit(1)

        n_done += len(x)

        x, x_mask, y_l, y_mask_l = prepare_data(x, y,
                                            n_words_src=options['n_words_src'],
                                            n_words=options['n_words'])

        ### in optional save weights mode.
        if alignweights:
            #TODO: this part does not work
            pprobs, attention = f_log_probs(*([x, x_mask]+ y_l + y_mask_l))[index]
            for jdata in get_alignments(attention, x_mask, y_mask):
                alignments_json.append(jdata)
        else:
            pprobs = f_log_probs(*([x, x_mask]+ y_l + y_mask_l))[index]

        # normalize scores according to output length
        if normalize:
            lengths = numpy.array([numpy.count_nonzero(s) for s in y_mask[0].T])
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
          factors=1, # input factors
          factors_tl=1, #output factors
          dim_per_factor=None, # list of word vector dimensionalities (one per factor): [250,200,50] for total dimensionality of 500
          dim_per_factor_tl=None,
          dim_word_per_factor_tl=None,
          lambda_parameter=0.3,
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
          n_words=None,  # target vocabulary size. List that contains one value per TL factor
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
          maxibatch_size=20,#How many minibatches to load at one time
          myinversegenerationdict=None,
          do_not_train_surface=False, # if enabled, cost function will not depend on the surface form layer
          finetune_surface_generator=False):


    # Model options
    model_options = locals().copy()

    if model_options['dim_per_factor'] == None:
        if factors == 1:
            model_options['dim_per_factor'] = [model_options['dim_word']]
        else:
            sys.stderr.write('Error: if using factored input, you must specify \'dim_per_factor\'\n')
            sys.exit(1)

    if model_options['dim_word_per_factor_tl'] == None:
        if factors_tl == 1:
            model_options['dim_word_per_factor_tl'] = [model_options['dim_word']]
        else:
            sys.stderr.write('Error: if using factored output, you must specify \'dim_word_per_factor_tl\'\n')
            sys.exit(1)

    if model_options['dim_per_factor_tl'] == None:
        if factors_tl == 1:
            model_options['dim_per_factor_tl'] = [model_options['dim']]
        else:
            sys.stderr.write('Error: if using factored output, you must specify \'dim_per_factor_tl\'\n')
            sys.exit(1)

    assert(len(dictionaries) == factors + factors_tl) # one dictionary per source factor + 1 for target factor
    assert(len(model_options['dim_per_factor']) == factors) # each factor embedding has its own dimensionality
    assert(len(model_options['dim_word_per_factor_tl']) == factors_tl) # each factor embedding has its own dimensionality
    assert(len(model_options['dim_per_factor_tl']) == factors_tl) # each decoder hidden state has its own dimensionality
    assert(sum(model_options['dim_per_factor']) == model_options['dim_word']) # dimensionality of factor embeddings sums up to total dimensionality of input embedding vector

    # load dictionaries and invert them
    worddicts = [None] * len(dictionaries)
    worddicts_r = [None] * len(dictionaries)
    for ii, dd in enumerate(dictionaries):
        worddicts[ii] = load_dict(dd)
        worddicts_r[ii] = dict()
        for kk, vv in worddicts[ii].iteritems():
            worddicts_r[ii][vv] = kk

    inversegenerationdict_d=dict()
    with open(myinversegenerationdict, 'rb') as f:
        inversegenerationdict_d=json.load(f)

    #n_words is now a list of lengths, one for each TL factor
    if n_words_src is None:
	n_words_src = len(worddicts[0])
        model_options['n_words_src'] = n_words_src
    if n_words is None:
        n_words=[]
    for factor in xrange(factors_tl):
        if len(n_words) < factor+1:
            n_words.append(len(worddicts[factors+factor]))
    model_options['n_words'] = n_words

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'Reloading model options'
        try:
            with open('%s.json' % saveto, 'rb') as f:
                loaded_model_options = json.load(f)
        except:
            with open('%s.pkl' % saveto, 'rb') as f:
                loaded_model_options = pkl.load(f)
        if "do_not_train_surface" in loaded_model_options:
            del loaded_model_options['do_not_train_surface']
        if "finetune_surface_generator" in loaded_model_options:
            del loaded_model_options['finetune_surface_generator']

        model_options.update(loaded_model_options)

    print 'Loading data'
    domain_interpolation_cur = None
    if use_domain_interpolation:
        print 'Using domain interpolation with initial ratio %s, increase rate %s' % (domain_interpolation_min, domain_interpolation_inc)
        domain_interpolation_cur = domain_interpolation_min
        train = DomainInterpolatorTextIterator(datasets[0], datasets[1],
                         dictionaries[:factors], dictionaries[factors:],
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
                         dictionaries[:factors], dictionaries[factors:],
                         n_words_source=n_words_src, n_words_target=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen,
                         shuffle_each_epoch=shuffle_each_epoch,
                         sort_by_length=sort_by_length,
                         maxibatch_size=maxibatch_size)

    if valid_datasets and validFreq:
        valid = TextIterator(valid_datasets[0], valid_datasets[1],
                            dictionaries[:factors], dictionaries[factors:],
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

    trng, use_noise, \
        x, x_mask, y_l, y_mask_l, \
        opt_ret, \
        cost, cost_surface, cost_factors_l = \
        build_model(tparams, model_options)

    #inps = [x, x_mask, y, y_mask]
    inps=[x, x_mask]
    inps.extend(y_l)
    inps.extend(y_mask_l)

    if (validFreq or sampleFreq) and not do_not_train_surface:
        print 'Building sampler'
        f_init, f_next = build_sampler(tparams, model_options, use_noise, trng)

    # before any regularizer
    print 'Building f_log_probs...',
    #f_log_probs = theano.function(inps, cost, profile=profile)
    #f_log_probs_surface=theano.function(inps, cost_surface, profile=profile)
    #f_log_probs_factors_l=[theano.function(inps, c, profile=profile) for c in cost_factors_l]
    if do_not_train_surface:
        f_log_probs_multi=theano.function(inps, [cost]+cost_factors_l, profile=profile)
    else:
        f_log_probs_multi=theano.function(inps, [cost,cost_surface]+cost_factors_l, profile=profile)
    print 'Done'


    cost = cost.mean()

    if not do_not_train_surface:
        cost_surface=cost_surface.mean()
    cost_factors_l=[ c.mean() for c in cost_factors_l ]

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
        for factor in xrange(factors_tl):
            alpha_reg = alpha_c * (
                (tensor.cast(y_mask_l[factor].sum(0)//x_mask.sum(0), 'float32')[:, None] -
                 opt_ret[ factored_layer_name('dec_alphas',factor)].sum(0))**2).sum(1).mean()
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

    ## Compile functions for cost of different factors
    #f_cost_surface = theano.function(inps, cost_surface, profile=profile)
    #f_cost_factors_l =[ theano.function(inps, cost_factor, profile=profile) for cost_factor in cost_factors_l]

    # allow finetuning with fixed embeddings
    if finetune:
        updated_params = OrderedDict([(key,value) for (key,value) in tparams.iteritems() if not key.startswith('Wemb')])
    else:
        updated_params = tparams

    # allow finetuning of only last layer (becomes a linear model training problem)
    if finetune_only_last:
        allowedParams=[]
        for factor in xrange(factors_tl):
            allowedParams.extend([ factored_layer_name('ff_logit',factor)+"_W", factored_layer_name('ff_logit',factor)+"_b"  ])
        #updated_params = OrderedDict([(key,value) for (key,value) in tparams.iteritems() if key in [ 'ff_logit_W', 'ff_logit_b']  ])
        updated_params = OrderedDict([(key,value) for (key,value) in tparams.iteritems() if key in allowedParams  ])
    elif finetune_surface_generator:
        myfactor=factors_tl-1
        updated_params = OrderedDict([(key,value) for (key,value) in tparams.iteritems() if key.startswith(embedding_name(myfactor)+'_dec') or key.startswith(factored_layer_name('ff_state',myfactor)) or key.startswith("rnngenerator") or key.startswith(factored_layer_name('ff_logit_lstm',myfactor)) or key.startswith(factored_layer_name('ff_logit_prev',myfactor)) or key.startswith(factored_layer_name('ff_logit_ctx',myfactor)) or key.startswith(factored_layer_name('ff_logit',myfactor)) ])
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

            if len(y) and len(y[0]) and len(y[0][0]) != factors_tl:
                sys.stderr.write('Error: mismatch between number of TL factors in settings ({0}), and number in training corpus ({1})\n'.format(factors_tl, len(y[0][0])))
                sys.exit(1)

            x, x_mask, y_l, y_mask_l = prepare_data(x, y, maxlen=maxlen,
                                                n_words_src=n_words_src,
                                                n_words=n_words)

            #print >>sys.stderr, "Obtained y_l from Minibatch: "+str(y_l)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            myinps=[x,x_mask]+y_l+y_mask_l
            #cost = f_grad_shared(x, x_mask, y_l, y_mask_l)
            cost = f_grad_shared(*myinps)

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
                GPUFreeMemoryInBytes = sbcuda.cuda_ndarray.cuda_ndarray.mem_info()[0]
                freeGPUMemInGBs = GPUFreeMemoryInBytes/1024./1024/1024
                #Compute cost of surface forms, cost of each factor, but only we want to display them (save time)
                #batchcost_surface = f_cost_surface(*myinps).mean()
                #batchcost_factors=[ f_cost_factor(*myinps).mean() for f_cost_factor in f_cost_factors_l  ]
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud, 'Free ',freeGPUMemInGBs
                #print 'Surface cost',batchcost_surface, 'Factors cost'," ".join([str(c) for c in batchcost_factors])

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
                                               x[:, :, jj][:, :, None],factors_tl=factors_tl,
                                               trng=trng, k=1,
                                               maxlen=30,
                                               stochastic=stochastic,
                                               argmax=False,
                                               suppress_unk=False,inversegenerationdict=inversegenerationdict_d)
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
                    factor=factors_tl-1
                    print 'Truth ', jj, ' : ',
                    for vv in y_l[factor][:, jj]:
                        if vv == 0:
                            break
                        if vv in worddicts_r[factors+factortl]:
                            print worddicts_r[factors+factortl][vv],
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
                        if vv in worddicts_r[factors+factortl]:
                            print worddicts_r[factors+factortl][vv],
                        else:
                            print 'UNK',
                    print

            # validate model on validation set and early stop if necessary
            if valid and validFreq and numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                curFLogIndex=0
                valid_errs, alignment = pred_probs_index(f_log_probs_multi, prepare_data,
                                        model_options, valid,curFLogIndex)
                valid_err =  valid_errs.mean()
                curFLogIndex+=1

                if not do_not_train_surface:
                    valid_errs_surface, alignment_surface = pred_probs_index(f_log_probs_multi, prepare_data,
                                            model_options, valid,curFLogIndex)
                    valid_err_surface =  valid_errs_surface.mean()
                    curFLogIndex+=1

                valid_err_factors_l=[]
                for c in cost_factors_l:#just iterating over tl factors
                    valid_errs_factor, alignment_factor = pred_probs_index(f_log_probs_multi, prepare_data,
                                            model_options, valid,curFLogIndex)
                    valid_err_factor =  valid_errs_factor.mean()
                    valid_err_factors_l.append(valid_err_factor)
                    curFLogIndex+=1

                #sum validation error for each factor
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
                if not do_not_train_surface:
                    print 'Valid_surface ', valid_err_surface
                print 'Valid_factors ', " ".join([str(valid_err_factor) for valid_err_factor in valid_err_factors_l])

                if external_validation_script:
                    print "Calling external validation script"
                    print 'Saving  model...',
                    params = unzip_from_theano(tparams)
                    numpy.savez(saveto +'.dev.'+str(uidx), history_errs=history_errs, uidx=uidx, **params)
                    jsonFile=saveto+'.dev.'+str(uidx)+'.npz.json'
                    json.dump(model_options, open( jsonFile, 'wb'), indent=2)
                    print '(not) Calling at '+time.strftime("%c")
                    #p = Popen([external_validation_script])

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

    #TODO fix me
    if valid:
        #use_noise.set_value(0.)
        #valid_errs, alignment = pred_probs(f_log_probs, prepare_data,
        #                                model_options, valid)
        #valid_err =  valid_errs.mean()

        use_noise.set_value(0.)
        curFLogIndex=0
        valid_errs, alignment = pred_probs_index(f_log_probs_multi, prepare_data,
                                model_options, valid,curFLogIndex)
        valid_err =  valid_errs.mean()
        curFLogIndex+=1

        if not do_not_train_surface:
            valid_errs_surface, alignment_surface = pred_probs_index(f_log_probs_multi, prepare_data,
                                    model_options, valid,curFLogIndex)
            valid_err_surface =  valid_errs_surface.mean()
            curFLogIndex+=1

        valid_err_factors_l=[]
        for c in cost_factors_l:#just iterating over tl factors
            valid_errs_factor, alignment_factor = pred_probs_index(f_log_probs_multi, prepare_data,
                                    model_options, valid,curFLogIndex)
            valid_err_factor =  valid_errs_factor.mean()
            valid_err_factors_l.append(valid_err_factor)
            curFLogIndex+=1

        print 'Valid ', valid_err
        if not do_not_train_surface:
            print 'Valid_surface ', valid_err_surface
        print 'Valid_factors ', " ".join([str(valid_err_factor) for valid_err_factor in valid_err_factors_l])


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
