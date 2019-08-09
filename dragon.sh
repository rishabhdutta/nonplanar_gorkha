#!/bin/bash

#SBATCH --job-name=Toh_main
#SBATCH --time=800:00
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1
#SBATCH --constraint=intel
#SBATCH --mail-user=rishabh.dutta@kaust.edu.sa
#SBATCH --mail-type=ALL

# Print some job information
echo
echo "My execution hostname is: $(hostname -s)."
echo "I am job $SLURM_JOB_ID, a member of the job array $SLURM_ARRAY_JOB_ID"
echo "and my task ID is $SLURM_ARRAY_TASK_ID"
echo

module purge all 
module load python/3.6.2

# check which stage is running 
stage=`ls samples1 | wc -l`
echo "stage $stage is running"

cd samples1/stage$stage
echo "changed directory to the present running stage"
numsamples=`ls samples | wc -l`

# get all the samples
while [ "$numsamples" -ne 2000 ]
do
	if [ "$numsamples" -lt 500 ]; then
		cd ../..
		echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
		echo "submitting sbatch to the cluster"
		echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
		start=`date +%s`
		rm slurm*
		#sh close_processes.sh
		sbatch python_run_intel.sh
		cd samples123/stage$stage
		sleep 14400s
		break
	fi
	ls samples/* > allind.txt
	cp ../../findmissing.py .
	python findmissing.py

	cd ../..
	cp samples1/stage$stage/vartorun.txt .
	cp samples1/stage$stage/endnum.txt .
	
	foo=`cat vartorun.txt`
	sed -i '9s/.*/'"$foo"'/' matlab_run_other.sh
	
	echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
	echo "submitting sbatch to the cluster"
	echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
	start=`date +%s`
	rm slurm* 
	#sh close_processes.sh
	sbatch python_run_other.sh

	cd samples1/stage$stage
	sleep 30s
	foo1=`cat endnum.txt`
	a="sample$stage"
	b="stage$foo1"
	c="new.mat"
	file=$a$b$c
	cd samples
	
	echo "----------checking the final file-------------------"
	
	while [ ! -e "$file" ]
	do
		end=`date +%s`
		runtime=$((end-start))
		
		if [ "$runtime" -gt 6000 ]; then
			break
		fi

		sleep 30s
	done
	cd ..
	numsamples=`ls samples | wc -l`
done


a=sample$stage
b=stage.mat
filename=$a$b
if [ ! -e "$filename" ]; then
	cp ../../combine_samps.py . 
	echo "------combining the samples--------"
	python combine_samps.py

	#echo "------plotting the results----------"
	#cp ../../plotslip.m .
	
fi


cd ../..
echo "in the folder '`pwd`'"

a="varname1 = 'samples1\/stage$stage\/sample$stage"
b="stage.mat'"
foo=$a$b
sed -i '131s/.*/'"$foo"'/' resample_stage.py
stagen=$((stage + 1))
mkdir samples1/stage$stagen
a="varname2 = 'samples1\/stage$stagen\/resampstage.mat'"
foo=$a
sed -i '153s/.*/'"$foo"'/' resample_stage.py

python resample_stage.py

a="varname1 = 'samples1\/stage$stagen\/resampstage.mat'"
sed -i '131s/.*/'"$a"'/' cont_sampleMH.py
a="varnamepart = 'samples1\/stage$stagen\/samples\/sample$stagen"
b="stage'"
foo=$a$b
sed -i '154s/.*/'"$foo"'/' cont_sampleMH.py

cd samples1/stage$stagen
mkdir samples
cd ../..

echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
echo "submitting sbatch to the cluster"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
start=`date +%s`
rm slurm*
#sh close_processes.sh
sbatch python_run_intel.sh

sleep 14400s
sbatch dragon.sh


