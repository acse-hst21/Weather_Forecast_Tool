from pkg_resources import get_distribution, DistributionNotFound

from .ConvLSTM import *  # noqa
from .ConvLSTMCell import *  # noqa
from .Seq2Seq import *  # noqa
from .StormDataset import *  # noqa
from .helper import *  # noqa

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
