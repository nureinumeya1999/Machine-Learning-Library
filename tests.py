from NeuralNetStructure import nnet, accuracy, activation, dropout, imagetools, \
    layer, loss, optimizers, pool
from preprocessor import Preprocessor
import numpy as np

input1 = np.array([

    [[[4, 0, 3, 4, 0],
      [1, 1, 4, 4, 1],
      [1, 3, 0, 2, 2],
      [3, 3, 4, 4, 1],
      [4, 1, 2, 1, 1]],

     [[3, 3, 1, 3, 3],
      [3, 0, 1, 4, 3],
      [3, 3, 2, 0, 1],
      [2, 1, 1, 1, 4],
      [1, 0, 3, 3, 3]],

     [[4, 2, 3, 4, 1],
      [1, 1, 3, 2, 3],
      [3, 2, 4, 2, 2],
      [1, 3, 1, 2, 2],
      [2, 2, 4, 3, 0]]],

    [[[1, 0, 2, 1, 1],
      [0, 1, 1, 0, 3],
      [2, 1, 1, 2, 1],
      [1, 1, 4, 4, 0],
      [2, 4, 0, 4, 3]],

     [[2, 0, 3, 4, 2],
      [3, 3, 3, 4, 2],
      [0, 4, 3, 4, 4],
      [4, 0, 3, 0, 1],
      [1, 1, 2, 0, 0]],

     [[2, 4, 0, 1, 2],
      [4, 4, 4, 1, 2],
      [0, 3, 0, 1, 1],
      [4, 3, 2, 2, 0],
      [2, 3, 0, 1, 2]]],

    [[[0, 0, 1, 1, 3],
      [1, 3, 1, 3, 1],
      [2, 2, 1, 4, 3],
      [3, 3, 2, 2, 2],
      [4, 0, 4, 4, 1]],

     [[2, 1, 0, 1, 3],
      [0, 2, 1, 0, 4],
      [2, 3, 1, 2, 3],
      [4, 0, 4, 0, 4],
      [2, 4, 4, 1, 1]],

     [[3, 2, 4, 3, 4],
      [4, 0, 1, 3, 4],
      [0, 3, 0, 1, 2],
      [0, 2, 3, 0, 1],
      [0, 3, 1, 3, 1]]],

    [[[4, 0, 3, 4, 2],
      [2, 0, 0, 0, 4],
      [4, 3, 3, 3, 2],
      [0, 4, 4, 3, 1],
      [0, 1, 1, 0, 0]],

     [[0, 4, 0, 2, 2],
      [1, 3, 2, 4, 0],
      [0, 1, 2, 1, 4],
      [1, 0, 0, 0, 3],
      [2, 3, 2, 2, 3]],

     [[0, 1, 3, 0, 2],
      [4, 0, 2, 1, 1],
      [1, 3, 4, 1, 2],
      [3, 0, 0, 4, 4],
      [0, 2, 0, 1, 2]]]])


def test_Pool_forward():
    a = pool.Pool(pool_dims=(2, 2), stride=(2, 2))

    a.forward(input1, True)
    actual = a.output
    expected = np.array([

        [[[4, 4],
          [3, 4]],

         [[3, 4],
          [3, 2]],

         [[4, 4],
          [3, 4]]],

        [[[1, 2],
          [2, 4]],

         [[3, 4],
          [4, 4]],

         [[4, 4],
          [4, 2]]],

        [[[3, 3],
          [3, 4]],

         [[2, 1],
          [4, 4]],

         [[4, 4],
          [3, 3]]],

        [[[4, 4],
          [4, 4]],

         [[4, 4],
          [1, 2]],

         [[4, 3],
          [3, 4]]]])
    assert ((actual == expected).all())
    assert ((actual.shape == (4, 3, 2, 2)))


input2 = np.array([[1, 2, 3, 4], [1, 2, 3, 2], [1, 2, 1, 1]])


def test_Activation_Softmax_forward():
    a = activation.Activation_Softmax()
    a.forward(input2, True)
    actual = a.output
    expected = np.array([[-3, -2, -1, 0], [-2, -1, 0, -1], [-1, 0, -1, -1]])
    expected = np.exp(expected)
    sums = np.sum(expected, axis=1, keepdims=True)
    expected = expected / sums
    assert ((actual == expected).all())


def test_Accuracy_Categorical_Calculate():
    a = accuracy.Accuracy_Categorical()
    b = activation.Activation_Softmax()
    a.new_pass()
    ground_truths = np.array([0, 2, 1])

    actual = a.calculate(b.predictions(input2), ground_truths)
    arg_max_input = np.argmax(input2, axis=1, keepdims=False)
    expected = np.mean(arg_max_input == ground_truths)

    assert ((actual == expected))


def test_Loss_Categorical_Crossentropy_Forward():
    a = loss.Loss_CategoricalCrossentropy()

    b = activation.Activation_Softmax()
    b.forward(input2, True)
    softmax_output = np.clip(b.output, 1e-7, 1 - 1e-7)

    ground_truths = np.array([0, 2, 1])
    actual = a.forward(softmax_output, ground_truths)

    expected = np.stack(
        [-np.log(softmax_output[0][0]), - np.log(softmax_output[1][2]),
         - np.log(softmax_output[2][1])])
    assert ((actual == expected).all())


