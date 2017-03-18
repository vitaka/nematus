import sys, pickle,json


#ARGUMENTS:
#- stdin: TL corpus file from which the generation dicrionary is going to be built
# sys.argv[1]: maximum vocabulary size for each factor, split by comma (,). "-" means no limit
# sys.argv[2:] one .json dictionary for each factor
# stodout: pickled dictionary

#json loads strings as unicode; we currently still work with Python 2 strings, and need conversion
def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        print >> sys.stderr, "Error opening dictionary"
        exit(1)

class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)

#load vocabulary size per factor
vocSizePerFactor =  [ int(limit)  if limit != "-" else None for limit in sys.argv[1].split(",") ]

#load json dictionaries
dictionaries = [ load_dict( sys.argv[2+i] ) for i in xrange(len(vocSizePerFactor)) ]

#from surface forms to set of tuples of factors. All of them encoded as numbers
generation_dict=dict()

#process corpus
for line in sys.stdin:
    line=line.strip().rstrip("\n")
    toks = line.split(" ")
    for tok in toks:
        factors=tok.split("|")
        if len(factors) != len(vocSizePerFactor):
            print >> sys.stderr, "Error: number of factors in token {0} do not match arguments ({1})".format(tok,len(vocSizePerFactor))
            exit(1)
        surface=dictionaries[-1][factors[-1]]
        if vocSizePerFactor[-1] == None or surface < vocSizePerFactor[-1]:
            if surface not in generation_dict:
                generation_dict[surface]=set()
            pre_analysis=tuple([ dictionaries[i][factors[i]] for i in xrange(len(vocSizePerFactor)-1) ])
            analysis=tuple([ pre_analysis[i] if ( vocSizePerFactor[i] == None or  pre_analysis[i] < vocSizePerFactor[i] ) else 1  for i in xrange(len(pre_analysis)) ]) # UNK = 1
            generation_dict[surface].add(analysis)

#write resulting reverse generation dictionary
#pickle.dump(generation_dict,sys.stdout)
json.dump(generation_dict, sys.stdout, indent=2, ensure_ascii=False, cls=SetEncoder)
