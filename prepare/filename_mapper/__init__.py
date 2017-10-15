import os

from . import filename_mapper


here = os.path.dirname(__file__)
mapper = filename_mapper.FilenameMapper(os.path.join(here, 'cache.pkl'))
