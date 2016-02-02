import data_nextchar

import numpy as np
import theano
import theano.tensor as T
import sys
import json

def dumpVec(vec):
    assert len(vec.shape) == 1
    return [float(t) for t in vec]

def dumpMat(m):
    assert len(m.shape) == 2
    return [[float(t) for t in line] for line in m]


class inputLayer:
    def __init__(self, numInputs):
        self.numInputs = numInputs
        self._output = T.dmatrix('input')
    def numOutputs(self):
        return self.numInputs
    def output(self):
        return self._output
    def parameters(self):
        return []
    def all_layers_with_params(self):
        return []
    def dump(self):
        return {'type': 'inputLayer',
                'numInputs': self.numInputs}

class layer:
    def __init__(self, below, numUnits, rng):
        self.below = below
        self.numUnits = numUnits
        self.rng = rng
        dataSize = below.numOutputs()
        maxw = 0.2
        self.initial_weights = np.minimum(
            maxw,
            np.maximum(
                -maxw,
                np.asarray(
                    rng.normal(scale=maxw/3,
                               size=(dataSize, numUnits)))))
        self.initial_bias = np.asarray(
                rng.uniform(low=0.0, high=0.0000000001,
                            size=(numUnits)))
        self.bias = theano.shared(self.initial_bias)
        self.weights = theano.shared(self.initial_weights)
        self.dropout = None
        self.theano_rng = None
        self.scaleWeights = None
    def numOutputs(self):
        return self.numUnits
    def output(self):
        input = self.below.output()
        weights = self.weights
        if self.scaleWeights != None:
            weights *= self.scaleWeights
        activations = T.dot(input, weights) + self.bias
        if self.dropout != None:
            if self.theano_rng == None:
                self.theano_rng = RandomStreams(self.rng.randint(2 ** 30))
            activeUnits = self.theano_rng.binomial(size=[self.numUnits],
                                                   n=1,
                                                   p=1.0-self.dropout)
            activations = activations * activeUnits
        return activations

    def regularization(self, f):
        return f*(T.sum(self.weights**2, axis=None)
                  + T.sum(self.bias**2, axis=None))
    def parameters(self):
        return [self.weights, self.bias]
    def all_layers_with_params(self):
        b = self.below.all_layers_with_params()
        b.append(self)
        return b
    def dump(self):
        return {'type': 'layer',
                'below': self.below.dump(),
                'numUnits': self.numUnits,
                'weights': dumpMat(self.weights.get_value()),
                'bias': dumpVec(self.bias.get_value())}
    def setWeightsAndBias(self, weights, bias):
        self.initial_weights = None
        self.initial_bias = None
        self.weights = theano.shared(np.array(weights))
        self.bias = theano.shared(np.array(bias))
    def setDropout(self, p):
        self.dropout = p
    def setScaleWeights(self, f):
        self.scaleWeights = f

class relu:
    def __init__(self, below):
        self.below = below
    def numOutputs(self):
        return self.below.numOutputs()
    def output(self):
        input = self.below.output()
        return T.maximum(0, input)
    def all_layers_with_params(self):
        return self.below.all_layers_with_params()
    def dump(self):
        return {'type': 'relu',
                'below': self.below.dump()}

class maxout:
    def __init__(self, below, n):
        self.below = below
        self.n = n
        self._numOutputs = int(self.below.numOutputs() / self.n)
    def numOutputs(self):
        return self._numOutputs
    def output(self):
        input = self.below.output()
        bs = input.shape[0]
        shapedInput = input.reshape([bs, self.numOutputs(), self.n])
        return T.max(shapedInput, axis=2)
    def all_layers_with_params(self):
        return self.below.all_layers_with_params()
    def dump(self):
        return {'type': 'maxout',
                'n': self.n,
                'below': self.below.dump()}
    
class softmax:
    def __init__(self, below):
        self.below = below
    def numOutputs(self):
        return self.below.numOutputs()
    def output(self):
        input = self.below.output()
        return T.nnet.softmax(input)
    def all_layers_with_params(self):
        return self.below.all_layers_with_params()
    def dump(self):
        return {'type': 'softmax',
                'below': self.below.dump()}

class negative_log_likelihood:
    def __init__(self, below, target):
        self.below = below
        self.target = target
    def numOutputs(self):
        return 1
    def output(self):
        input = self.below.output()
        s = self.target.shape[0]
        ch = T.log(input)[T.arange(s), self.target]
        return -T.mean(ch)
    def all_layers_with_params(self):
        return self.below.all_layers_with_params()
    def dump(self):
        return {'type': 'negative_log_likelihood',
                'below': self.below.dump()}

def rebuild(d):
    rng = np.random.RandomState(123)
    t = d['type']
    if t == 'inputLayer':
        return inputLayer(d['numInputs'])
    elif t == 'layer':
        l = layer(rebuild(d['below']), d['numUnits'], rng)
        l.setWeightsAndBias(d['weights'], d['bias'])
        return l
    elif t == 'relu':
        return relu(rebuild(d['below']))
    elif t == 'maxout':
        n = 4
        if 'n' in d:
            n = d['n']
        return maxout(rebuild(d['below']), n)
    elif t == 'softmax':
        return softmax(rebuild(d['below']))
    elif t == 'negative_log_likelihood':
        target = T.ivector('target')
        return negative_log_likelihood(rebuild(d['below']), target)

    
def updateFunction(input, target, error, layers, lr):
    params = [p for l in layers for p in l.parameters()]
    grad = T.grad(error, params)
    updates = [(params[i], params[i] - lr*grad[i])
               for i in range(len(params))]
    return theano.function([input, target, lr],
                           error,
                           updates=updates)

def testFunction(input, target, nll, softmax, layers):
    return theano.function(
        [input, target],
        [nll, softmax.output()] + [l.output() for l in layers])

def r3(x):
    return round(x*1000)*0.001

def f3(x):
    return '{:1.3f}'.format(x)


def loadNet(filename):
    with open(filename, 'r') as file:
        return rebuild(json.loads(file.read()))

def getInputLayer(l):
    t = type(l)
    if t == inputLayer:
        return l
    elif (t == layer
          or t == relu
          or t == maxout
          or t == softmax
          or t == negative_log_likelihood):
        return getInputLayer(l.below)
    else:
        print('unrecog layer', l)
        assert False

def allLayerActivationFunctions(l):
    t = type(l)
    if (t == relu
        or t == maxout):
        return allLayerActivationFunctions(l.below) + [l]
    elif t == inputLayer:
        return []
    elif (t == layer
          or t == softmax
          or t == negative_log_likelihood):
        return allLayerActivationFunctions(l.below)
    else:
        print('unrecog layer', l)
        assert False

def allUnitsLayers(l):
    t = type(l)
    if t == layer:
        return allUnitsLayers(l.below) + [l]
    elif t == inputLayer:
        return []
    elif (t == relu
          or t == softmax
          or t == maxout
          or t == negative_log_likelihood):
        return allUnitsLayers(l.below)
    else:
        print('unrecog layer', l)
        assert False

def dumpNetworkStructure(l):
    t = type(l)
    if t == layer:
        dumpNetworkStructure(l.below)
        if l.dropout != None:
            print('layer', l.numUnits, 'dropout', l.dropout)
        else:
            print('layer', l.numUnits)
    elif t == inputLayer:
        print('inputLayer', l.numInputs)
    elif t == relu:
        dumpNetworkStructure(l.below)
        print('relu')
    elif t == maxout:
        dumpNetworkStructure(l.below)
        print('maxout', l.n)
    elif t == softmax:
        dumpNetworkStructure(l.below)
        print('softmax')
    elif t == negative_log_likelihood:
        dumpNetworkStructure(l.below)
        print('negative_log_likelihood')
    else:
        print('unrecog layer', l)
        assert False

def setDropoutToAllUnits(net, p, scaleWeights):
    for l in allUnitsLayers(net):
        l.setDropout(p)
        l.setScaleWeights(scaleWeights)


def learn(filename):
    data_nextchar.init()
    context_size = 7
    if filename == None:
        rng = np.random.RandomState(123)
        input = inputLayer(data_nextchar.input_width(context_size))
        l1 = layer(input, 2000, rng)
        l1o = maxout(l1, 4)
        l2 = layer(l1o, 800, rng)
        l2o = maxout(l2, 4)
        l3 = layer(l2o, 800, rng)
        l3o = maxout(l3, 4)
        l4 = layer(l3o, 800, rng)
        l4o = maxout(l4, 4)
        llast = layer(l4o, data_nextchar.numChars(), rng)
        output = softmax(llast)
        target_ = T.ivector('target')
        errorL = negative_log_likelihood(output, target_)
    else:
        errorL = loadNet(filename)
        input = getInputLayer(errorL)
        output = errorL.below
        llast = output.below

    target = errorL.target
    error = errorL.output()
    nll = negative_log_likelihood(output, target).output()

    lr = T.scalar('lr')
    setDropoutToAllUnits(errorL, 0.4, None)
    dumpNetworkStructure(errorL)
    trainFunc = updateFunction(
        input.output(),
        target,
        error,
        errorL.all_layers_with_params(),
        lr)

    setDropoutToAllUnits(errorL, None, 0.6)
    testFunc = testFunction(
        input.output(),
        target,
        nll,
        output,
        allLayerActivationFunctions(errorL))

    minibatchSize = 200

    testingMinibatchSize = 10000
    (testInputs, testOutputs, itxts, otxts) = data_nextchar.prepareMinibatch(
        testingMinibatchSize, context_size, False)

    trainingErrors = []
    t = 0
    while True:
        (inputs, outputs, tritxts, trotxts) = data_nextchar.prepareMinibatch(
            minibatchSize, context_size, True)
        err = trainFunc(inputs.astype(np.float32),
                        outputs.astype(np.int32),
                        0.003)
        trainingErrors.append(err)
        #print(t, '   training:', r3(err))
        if t<20 or (t<100 and t%10 == 0) or t%50 == 0:
            trainingErr = np.mean(trainingErrors)
            trainingErrors = []
        
            tres = testFunc(testInputs.astype(np.float32),
                            testOutputs.astype(np.int32))
            err = tres[0]
            relus = tres[2:]
            ### print('num >0  ', ', '.join([f3((r>0).sum()/r.size) for r in relus]))
            ### print('num >0.3', ', '.join([f3((r>.3).sum()/r.size) for r in relus]))
            ### print('num >1  ', ', '.join([f3((r>1).sum()/r.size) for r in relus]))
            #lo = tres[2]
            # print(lo[0,::])
            # print(lo[1,::])
            # print(lo[2,::])
            sm = tres[1]
            smo = np.argsort(sm, axis=1)[:, ::-1]
            #print(smo[1:3])
            hits = list([0 for _ in range(0, smo.shape[1])])
            for v in range(0, smo.shape[0]):
                for o in range(0, smo.shape[1]):
                    if smo[v,o] == testOutputs[v]:
                        hits[o] += 1
                        continue
            print(t, 'testing:', r3(err), 'training:', r3(trainingErr), hits)
            sys.stdout.flush()
            # for m in range(50):
            #     print(titxt[m], totxt[m],
            #           [charidToChar(smo[m,p]) + " {:9.7f}".format(sm[m, smo[m,p]])
            #            for p in range(5)])

        if t%1000 == 0:
            filename = 'netm.json'
            with open(filename, 'w') as f:
                f.write(json.dumps(errorL.dump()))
                print('wrote', filename)
        t += 1

def reportLargeWeights(filename):
    errorL = loadNet(filename)
    input = getInputLayer(errorL)
    output = errorL.below
    llast = output.below
    units = allUnitsLayers(errorL)
    first = units[0]
    print(first)
    wt = np.transpose(first.weights.get_value())
    bias = first.bias.get_value()
    print('wt shape', wt.shape)
    max = np.max(wt)
    min = np.min(wt)
    print('max', max, 'min', min)
    f = 0.5
    extreme = (wt > f*max) + (wt < f*min)
    nc = data_nextchar.numChars()
    for u in range(0, wt.shape[0]):
        print('unit', u, 'bias', bias[u])
        for i in np.argwhere(extreme[u,::]):
            c = i[0]
            nchar = c//nc
            ch = c % nc
            print(nchar, data_nextchar.charAt(ch), wt[u, i][0])


#learn(None)
reportLargeWeights('netm.json')
