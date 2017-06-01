import tensorflow as tf
import numpy as np

BATCH_START = 0
TIME_STEPS = 2
BATCH_SIZE = 5
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006
BATCH_START_TEST = 0

def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    print(xs)
    seq = np.sin(xs)
    
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
##    plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
##    plt.show()
    # returned seq, res and shape (batch, step, input)
    return seq, res, xs

seq, res, xs = get_batch()
print(seq.reshape((BATCH_SIZE, TIME_STEPS, INPUT_SIZE)))
xs = np.zeros((BATCH_SIZE, TIME_STEPS, OUTPUT_SIZE))
print(xs)