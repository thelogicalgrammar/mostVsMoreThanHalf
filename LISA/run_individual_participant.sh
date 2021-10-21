# Run from the main directory
for single_participant_data in ../data/* 
do
	sbatch job_single_participant.sh $single_participant_data
done
