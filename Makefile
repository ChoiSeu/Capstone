SHELL=/bin/sh

PYTHON		?= python3
SOURCE		 = src
MODEL		 = model
SAMPLE		 = test_sample

all: run

run run_test run.test test:
	${PYTHON} ${SOURCE}/test.py --model=${MODEL}.pth\
		--sample=${SAMPLE}.pcap
