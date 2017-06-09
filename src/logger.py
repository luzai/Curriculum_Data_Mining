import logging
import os
import sys

format = '%(asctime)s - %(filename)s - [line:%(lineno)d] - %(levelname)s - %(message)s'
formatter = logging.Formatter(format)

if os.path.exists('../output') == False:
    os.mkdir('../output')

infoLogName = '../output/svd.log'
infoLogger = logging.getLogger("infoLog")
infoLogger.setLevel(logging.INFO)
infoHandler = logging.FileHandler(infoLogName, 'w')
infoHandler.setLevel(logging.INFO)
infoHandler.setFormatter(formatter)
infoLogger.addHandler(infoHandler)

stdoutHandler = logging.StreamHandler(sys.stdout)
stdoutHandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s ==> %(message)s')
stdoutHandler.setFormatter(formatter)
infoLogger.addHandler(stdoutHandler)

logger = infoLogger
