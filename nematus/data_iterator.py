import numpy

import gzip

import shuffle
from util import load_dict

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, target,
                 source_dicts, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 use_factor=False,
                 maxibatch_size=20,
                 interleave_tl=False,
                 use_factor_tl=False,
                 n_words_target_factor1=-1):
        if shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source = fopen(source, 'r')
            self.target = fopen(target, 'r')
        self.source_dicts = []
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict))
        if use_factor_tl:
            self.target_dict = [load_dict(one_target_dict) for one_target_dict in target_dict ]
        else:
            self.target_dict = load_dict(target_dict)

        self.batch_size = batch_size
        self.maxlen_sl = maxlen
        self.maxlen_tl= maxlen
        self.interleave_tl=interleave_tl
        self.use_factor_tl=use_factor_tl
        self.skip_empty = skip_empty
        self.use_factor = use_factor

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target
        self.n_words_target_factor1=n_words_target_factor1

        if self.n_words_source > 0:
            for d in self.source_dicts:
                for key, idx in d.items():
                    if idx >= self.n_words_source:
                        del d[key]

        if self.n_words_target > 0:
            if use_factor_tl:
                for dictindex,d in enumerate(self.target_dict):
                    for key, idx in d.items():
                        if dictindex != 1:
                            compareWith=self.n_words_target
                        else:
                            compareWith=self.n_words_target_factor1
                        if compareWith >= 0 and idx >= compareWith :
                            del d[key]
            else:
                for key, idx in self.target_dict.items():
                    if idx >= self.n_words_target:
                        del self.target_dict[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size


        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])

    def reset(self):
        if self.shuffle:
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)

    def next(self):
        INTERLEAVE_PREFIX="interleaved_"
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for ss in self.source:
                ss = ss.split()
                tt = self.target.readline().split()

                if self.skip_empty and (len(ss) == 0 or len(tt) == 0):
                    continue

                if len(ss) > self.maxlen_sl:
                    continue

                if self.interleave_tl:
                    tt_data=[t for t in tt if not t.startswith(INTERLEAVE_PREFIX)]
                    if len(tt_data) > self.maxlen_tl:
                        continue
                else:
                    if len(tt) > self.maxlen_tl:
                        continue

                self.source_buffer.append(ss)
                self.target_buffer.append(tt)
                if len(self.source_buffer) == self.k:
                    break

            if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration

            # sort by target buffer
            if self.sort_by_length:
                tlen = numpy.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()


        try:
            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                tmp = []
                for w in ss:
                    if self.use_factor:
                        w = [self.source_dicts[i][f] if f in self.source_dicts[i] else 1 for (i,f) in enumerate(w.split('|'))]
                    else:
                        w = [self.source_dicts[0][w] if w in self.source_dicts[0] else 1]
                    tmp.append(w)
                ss = tmp

                # read from target file and map to word index
                tt = self.target_buffer.pop()
                if self.use_factor_tl:
                    tmp = []
                    for w in tt:
                        w = [self.target_dict[i][f] if f in self.target_dict[i] else 1 for (i,f) in enumerate(w.split('|'))]
                        tmp.append(w)
                    tt=tmp
                else:
                    tt = [self.target_dict[w] if w in self.target_dict else 1
                          for w in tt]
                    if self.n_words_target > 0:
                        tt = [w if w < self.n_words_target else 1 for w in tt]

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        return source, target
