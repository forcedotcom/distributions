import os
import numpy
import scipy
import scipy.misc
from distributions.dbg.random import sample_discrete, sample_discrete_log
from distributions.lp.models.nich import NormalInverseChiSq
from distributions.lp.clustering import PitmanYor
from distributions.io.stream import json_stream_load, json_stream_dump
import parsable
parsable = parsable.Parsable()


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(ROOT, 'results')
SAMPLES = os.path.join(DATA, 'samples.json.gz')
IMAGE = scipy.lena()


class ImageModel(object):
    def __init__(self):
        self.clustering = PitmanYor.model_load({
            'alpha': 1.0,
            'd': 0.2,
        })
        self.feature = NormalInverseChiSq.model_load({
            'mu': 0.0,
            'kappa': 1.0,
            'sigmasq': 1.0,
            'nu': 1.0,
        })

    class Mixture(object):
        def __init__(self):
            self.clustering = PitmanYor.Mixture()
            self.feature_x = NormalInverseChiSq.Mixture()
            self.feature_y = NormalInverseChiSq.Mixture()

        def __len__(self):
            return len(self.clustering)

        def init_empty(self, model):
            self.clustering.clear()
            self.feature_x.clear()
            self.feature_y.clear()

            # Add a single empty group
            self.clustering.append(0)
            self.feature_x.add_group(model.feature)
            self.feature_y.add_group(model.feature)

            self.clustering.init(model.clustering)
            self.feature_x.init(model.feature)
            self.feature_y.init(model.feature)

        def score_value(self, model, xy, scores):
            x, y = xy
            self.clustering.score(model.clustering, scores)
            self.feature_x.score_value(model.feature, x, scores)
            self.feature_y.score_value(model.feature, y, scores)

        def add_value(self, model, groupid, xy):
            x, y = xy
            group_added = self.clustering.add_value(model.clustering, groupid)
            self.feature_x.add_value(model.feature, groupid, x)
            self.feature_y.add_value(model.feature, groupid, y)
            if group_added:
                self.feature_x.add_group(model.feature)
                self.feature_y.add_group(model.feature)


def sample_from_2d_array(image, sample_count):
    x_pmf = image.sum(axis=1)
    y_pmfs = image.copy()
    for y_pmf in y_pmfs:
        y_pmf /= y_pmf.sum()

    x_scale = 2.0 / (image.shape[0] - 1)
    y_scale = 2.0 / (image.shape[1] - 1)

    for _ in xrange(sample_count):
        x = sample_discrete(x_pmf)
        y = sample_discrete(y_pmfs[x])
        yield (x * x_scale - 1.0, y * y_scale - 1.0)


def synthesize_2d_array(model, mixture):
    width, height = IMAGE.shape
    image = numpy.zeros((width, height))
    scores = numpy.zeros(len(mixture), dtype=numpy.float32)
    x_scale = 2.0 / (width - 1)
    y_scale = 2.0 / (height - 1)
    for x in xrange(width):
        for y in xrange(height):
            xy = (x * x_scale - 1.0, y * y_scale - 1.0)
            mixture.score_value(model, xy, scores)
            prob = numpy.exp(scores).sum()
            image[x, y] = prob
    image *= 255 / image.max()
    return image


@parsable.command
def create_dataset(sample_count=10000):
    '''
    Extract dataset from image.
    '''
    image = -1.0 * IMAGE
    image -= image.min()
    samples = sample_from_2d_array(image, sample_count)
    json_stream_dump(samples, SAMPLES)


@parsable.command
def compress_sequential():
    '''
    Compress image via sequential initialization.
    '''
    assert os.path.exists(SAMPLES), 'first create dataset'
    model = ImageModel()
    mixture = ImageModel.Mixture()
    mixture.init_empty(model)
    scores = numpy.zeros(1, dtype=numpy.float32)
    for xy in json_stream_load(SAMPLES):
        scores.resize(len(mixture))
        mixture.score_value(model, xy, scores)
        groupid = sample_discrete_log(scores)
        mixture.add_value(model, groupid, xy)
    print 'found {} components'.format(len(mixture))
    # TODO save model
    # TODO synthesize image


@parsable.command
def compress_gibbs(passes=100):
    '''
    Compress image via gibbs sampling.
    '''
    raise NotImplementedError()


@parsable.command
def compress_annealing(passes=100):
    '''
    Compress image via subsample annealing.
    '''
    raise NotImplementedError()


def test(sample_count=100):
    create_dataset(sample_count)
    compress_sequential()


if __name__ == '__main__':
    parsable.dispatch()
