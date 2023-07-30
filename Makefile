SHELL=/bin/sh

PYTHON		?= python3
SOURCE		 = src
MODEL		 = model
SAMPLE		 = test_sample

all:

run_test run.test test:
	${PYTHON} ${SOURCE}/test.py --model=${MODEL}.pth\
		--sample=${SAMPLE}.pcap

run_train run.train train:
	${PYTHON} ${SOURCE}/train.py

set:
	pip install csikit
