#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
connect_hpc:
	ssh s213160@login1.hpc.dtu.dk

enviroment:
	conda activate hpc_env

check_hpc:
	qstat

my_jobs:
	bstat

git_hard_reset:
	git reset --hard origin/main

## Install Python Dependencies
requirements: 
	pip install -U pip setuptools wheel
	pip install -r requirements.txt

## connect to wandb
wandb_connect:
	wandb login
## 5040f034163d656f6ef8acfcbf76f762e688e64f

##update req file
update_req:
	pipreqs . --force

#git pull
#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

#run job file
run:
	bsub < jobscript.sh

kill:
	bkill $(ID)

#run job file
test:
	bsub < test_jobscript.sh

#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