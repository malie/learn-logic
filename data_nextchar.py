import numpy as np
import theano
import theano.tensor as T
import random

def randint(n):
    return random.randint(0, n-1)

global wholeText
def readTextFile():
    global wholeText
    with open('pg76.txt', 'r') as textfile:
        wholeText = textfile.read().replace('\n', ' ').replace('\r', ' ')

global numChars, charmap, unknownChar
global characters
#characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ ,.:;\'"-()?! '
characters = 'ABCDEFGHILMNOPRSTU '

def numChars():
    return len(characters)+1

def charAt(n):
    if n < len(characters):
        return characters[n]
    else:
        return '%'

def initCharmap():
    global charmap, unknownChar, characters
    unknownChar = len(characters)
    charmap = dict(zip(characters, range(len(characters))))

def lookupChar(c):
    u = c.upper()
    if u in charmap:
        return charmap[u]
    else:
        return unknownChar

def charidToChar(c):
    if c < len(characters):
        return characters[c]
    else:
        return '%'

def input_width(contextSize):
    return numChars() * contextSize
        
def prepareMinibatch(minibatchSize, contextSize, forTraining):
    global wholeText
    inputs = np.zeros([minibatchSize, contextSize*numChars()])
    outputs = np.zeros([minibatchSize], dtype=np.int32)
    inputTexts = []
    outputTexts = []
    res = []
    for b in range(minibatchSize):
        while True:
            start = randint(len(wholeText)-contextSize)
            m = start % 100 < 90
            if m != forTraining:
                continue
            for i in range(contextSize):
                ch = lookupChar(wholeText[start+i])
                inputs[b, ch] = 1
            end = start + contextSize
            outputs[b] = lookupChar(wholeText[end])
            inputTexts.append(wholeText[start:end])
            outputTexts.append(wholeText[end:end+1])
            break
    return (inputs, outputs, inputTexts, outputTexts)

def init():
    readTextFile()
    initCharmap()
    
