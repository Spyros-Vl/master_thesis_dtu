#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
connect_hpc:
	ssh s213160@login1.hpc.dtu.dk

enviroment:
	conda activate hpc_env

check_hpc:
	qstat

## Install Python Dependencies
requirements: 
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
