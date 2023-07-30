SHELL=/bin/sh

PYTHON		?= python3
SOURCE		 = src

all:

SAMPLE		 = test_sample
TEST		 = model

run_test run.test test:
	${PYTHON} ${SOURCE}/test.py --model=${TEST}.pth\
		--sample=${SAMPLE}.pcap


DATE		 = 0725
NUM		 = 100
BATCH		 = 20
EPOCHS		 = 100
LEARNING_RATE	 = 0.001
GOAL		 = 0.9

run_train run.train train:
	${PYTHON} ${SOURCE}/train.py --date=${DATE}\
		--num=${NUM}\
		--batch_size=${BATCH}\
		--epochs=${EPOCHS}\
		--lr=${LEARNING_RATE}\
		--more=False\
		--goal=${GOAL}

TRAIN		 = checkpoint_acc_0.81

run_more run.more train_more train.more more:
	${PYTHON} ${SOURCE}/train.py --date=${DATE}\
		--num=${NUM}\
		--batch_size=${BATCH}\
		--epochs=${EPOCHS}\
		--lr=0.0001\
		--more=True\
		--goal=0.95\
		--pretrained_model=${TRAIN}.pth

set:
	pip install csikit
	conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
