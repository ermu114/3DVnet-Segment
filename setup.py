from distutils.core import setup
from Cython.Build import cythonize
import shutil,os
setup(name = 'data_processor', ext_modules = cythonize('./utility/data_processor.py'))
setup(name = 'parameters', ext_modules = cythonize('./utility/parameters.py'))
setup(name = 'iodicom', ext_modules = cythonize('./utility/iodicom.py'))
setup(name = 'predictor_interface', ext_modules = cythonize('./predict/predictor_interface.py'))
setup(name = 'dataset', ext_modules = cythonize('./data/dataset.py'))
setup(name = 'interface', ext_modules = cythonize('interface.py'))

if os.path.exists('./release/RAIContour/bin/Release/utility/data_processor.cp36-win_amd64.pyd'):
    os.remove('./release/RAIContour/bin/Release/utility/data_processor.cp36-win_amd64.pyd')
shutil.move('./build/lib.win-amd64-3.6/data_processor.cp36-win_amd64.pyd','./release/RAIContour/bin/Release/utility/')

if os.path.exists('./release/RAIContour/bin/Release/utility/parameters.cp36-win_amd64.pyd'):
    os.remove('./release/RAIContour/bin/Release/utility/parameters.cp36-win_amd64.pyd')
shutil.move('./build/lib.win-amd64-3.6/parameters.cp36-win_amd64.pyd','./release/RAIContour/bin/Release/utility/')

if os.path.exists('./release/RAIContour/bin/Release/utility/iodicom.cp36-win_amd64.pyd'):
    os.remove('./release/RAIContour/bin/Release/utility/iodicom.cp36-win_amd64.pyd')
shutil.move('./build/lib.win-amd64-3.6/iodicom.cp36-win_amd64.pyd','./release/RAIContour/bin/Release/utility/')

if os.path.exists('./release/RAIContour/bin/Release/predict/predictor_interface.cp36-win_amd64.pyd'):
    os.remove('./release/RAIContour/bin/Release/predict/predictor_interface.cp36-win_amd64.pyd')
shutil.move('./build/lib.win-amd64-3.6/predictor_interface.cp36-win_amd64.pyd','./release/RAIContour/bin/Release/predict/')

if os.path.exists('./release/RAIContour/bin/Release/data/dataset.cp36-win_amd64.pyd'):
    os.remove('./release/RAIContour/bin/Release/data/dataset.cp36-win_amd64.pyd')
shutil.move('./build/lib.win-amd64-3.6/dataset.cp36-win_amd64.pyd','./release/RAIContour/bin/Release/data/')

if os.path.exists('./release/RAIContour/bin/Release/interface.cp36-win_amd64.pyd'):
    os.remove('./release/RAIContour/bin/Release/interface.cp36-win_amd64.pyd')
shutil.move('./build/lib.win-amd64-3.6/interface.cp36-win_amd64.pyd','./release/RAIContour/bin/Release/')

if os.path.exists('./utility/data_processor.c'):
    os.remove('./utility/data_processor.c')
if os.path.exists('./utility/parameters.c'):
    os.remove('./utility/parameters.c')
if os.path.exists('./utility/iodicom.c'):
    os.remove('./utility/iodicom.c')
if os.path.exists('./predict/predictor_interface.c'):
    os.remove('./predict/predictor_interface.c')
if os.path.exists('./data/dataset.c'):
    os.remove('./data/dataset.c')
if os.path.exists('./interface.c'):
    os.remove('./interface.c')