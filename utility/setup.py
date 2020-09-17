from distutils.core import setup
from Cython.Build import cythonize
setup(name = 'dicomio',
      ext_modules = cythonize('iodicom.py'))
setup(name = 'common',
      ext_modules = cythonize('common.py'))
setup(name = 'filepath',
      ext_modules = cythonize('filepath.py'))