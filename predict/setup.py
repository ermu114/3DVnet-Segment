from distutils.core import setup
from Cython.Build import cythonize
setup(name = 'predictor',
      ext_modules = cythonize('predictor.py'))