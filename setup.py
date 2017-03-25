import sys

from Cython.Distutils import build_ext
from distutils.core import setup
from distutils.extension import Extension
import numpy


ex_cargs = []
ex_largs = []
if sys.platform.startswith('win'):
    ex_cargs.append('/openmp')
else:
    ex_cargs.append('-fopenmp')
    ex_largs.append('-fopenmp')

ext_modules = [
    Extension(
        '_filter',
        sources=['_filter.pyx'],
        include_dirs=[numpy.get_include()],
        extra_compile_args=ex_cargs,
        extra_link_args=ex_largs)
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext}
)
