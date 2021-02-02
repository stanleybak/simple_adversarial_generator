'''
adversarial generation using foolbox to find adversarial examples

requires input model in tensorflow1 format (.pb)

assumes input ranges are -1 to 1

Stanley Bak, Jan 2021
'''

import os
import logging
import time
import warnings

import numpy as np

import onnx

import tensorflow as tf
import foolbox as fb

from util import get_io_nodes

class AgenState():
    'adversarial image generation container'

    def __init__(self, onnx_filename, orig_image=None, epsilon=1.0, bounds=(-1.0, 1.0)):
        '''initialize a session'''

        warnings.filterwarnings("ignore", category=UserWarning)
        logging.basicConfig(level=logging.ERROR)
        #warnings.filterwarnings("ignore", message='exponential search failed')

        # turn of logging errors
        #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
        logging.getLogger('tensorflow').setLevel(logging.FATAL)

        # disable eager execution
        tf.compat.v1.disable_eager_execution()

        # slightly hack... onnx model is used only to get input / output names
        assert onnx_filename.endswith(".onnx")
        model = onnx.load(onnx_filename)

        inp, out = get_io_nodes(onnx_model)
    
        self.input_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)
        self.output_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in out.type.tensor_type.shape.dim)

        self.input_name = inp.name
        self.output_name = out.name

        filename = f'{onnx_filename}.pb'
        graph_def = None

        with tf.io.gfile.GFile(filename, 'rb') as f: 
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        assert graph_def is not None

        self.graph_def = graph_def

        if orig_image is None:
            self.orig_image = np.zeros(self.input_shape, dtype=np.float32)
        else:
            self.orig_image = orig_image
        
        self.epsilon = epsilon
        self.bounds = bounds

        self.sess = tf.compat.v1.Session()

        with self.sess.as_default():
            tf.import_graph_def(self.graph_def, name='')
            graph = self.sess.graph

            input_tensor = graph.get_tensor_by_name(self.input_name)
            output_tensor = graph.get_tensor_by_name(self.output_name)

            self.fmodel = fb.models.TensorFlowModel(input_tensor, output_tensor, bounds=self.bounds)

    def __del__(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None


    def try_single(self):
        '''try to generate an adversarial image for the single value of epsilon (quick)

        returns [adversarial image, epsilon], if found, else None
        '''

        rv = None

        criterion = fb.criteria.Misclassification()

        with self.sess.as_default():
            attack = SingleEpsilonRPGD(self.fmodel, distance=fb.distances.Linfinity, criterion=criterion)

            # subtract a small amount since attack was overshooting by numerical precision
            SingleEpsilonRPGD.set_epsilon(self.epsilon - 1e-9)

            a = attack(self.orig_image, self.labels, unpack=False)[0]

            dist = a.distance.value

            if dist != np.inf:
                rv = [a.perturbed, dist]
                rv[0].shape = self.orig_image.shape


        return rv


class SingleEpsilonRPGD(
    fb.attacks.iterative_projected_gradient.LinfinityGradientMixin,
    fb.attacks.iterative_projected_gradient.LinfinityClippingMixin,
    fb.attacks.iterative_projected_gradient.LinfinityDistanceCheckMixin,
    fb.attacks.iterative_projected_gradient.GDOptimizerMixin,
    fb.attacks.iterative_projected_gradient.IterativeProjectedGradientBaseAttack,
):
    'random projected gradient descent with custom parameters'

    epsilon = 0 # bad... storing in class variable

    @classmethod
    def set_epsilon(cls, epsilon):
        'set epsilon value'
        cls.epsilon = epsilon

    @fb.attacks.base.generator_decorator
    def as_generator(
        self,
        a,
        binary_search=False,
        epsilon=None, # use from constructor
        stepsize=0.01,
        iterations=50,
        random_start=True,
        return_early=True,
    ):
        epsilon = SingleEpsilonRPGD.epsilon
        assert epsilon > 0

        yield from self._run(
            a, binary_search, epsilon, stepsize, iterations, random_start, return_early
        )

def try_quick_adversarial(onnx_path, num_attempts, remaining_secs=None):
    '''try a quick adversarial example using Settings

    returns AgenState instance, aimage (may be None)
    '''

    start = time.perf_counter()

    agen = AgenState(onnx_path)
    a = None

    for i in range(num_attempts):

        if remaining_secs is not None:
            diff = time.perf_counter() - start
            
            if diff > remaining_secs:
                break # timeout!
        
        a = agen.try_single()

        if a is not None:
            break

    if a is not None:
        aimage, ep = a

        if Settings.PRINT_OUTPUT and num_attempts > 0:
            print(f"try_quick_adversarial found violation image on iteration {i} with ep={ep}")
        
    else:
        aimage = None

    #with agen.sess.as_default():
    #    attack = fb.attacks.RandomPGD(agen.fmodel, distance=fb.distances.Linfinity)

    #    Timers.tic('attack')
    #    a = attack(agen.orig_image, agen.labels, unpack=False)[0]
    #    Timers.toc('attack')

    #    dist = a.distance.value

    #    print(f"\nDist of random PGD adversarial: {dist}")
        
    return agen, aimage
