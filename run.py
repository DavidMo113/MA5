import sys
import os
import json

sys.path.insert(0, 'src')

from ma5 import data_batch
from ma5 import model_train

if 'data' in targets:
	with open('Test/testdata') as fh:
		data = data_batch(fh)

if 'model' in targets:
	model_train(data)


if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = 'data,model'
    main(targets)