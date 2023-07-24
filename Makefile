SHELL=/bin/sh

PYTHON		?= python3
MODEL		 = model
SAMPLE		 = test_sample

all:

run run_test run.test test:
	${PYTHON} test.py --model=${MODEL}.pth\
		--sample=${SAMPLE}.pcap
