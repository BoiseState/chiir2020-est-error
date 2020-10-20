#!/bin/sh
#SBATCH --exclusive -c28 -N1
#SBATCH -o build/sims_results.o%j 

echo running on $(hostname)

OUTPUTFILE=build/sims_results.o$SLURM_JOB_ID

echo "output is saved to $OUTPUTFILE"

echo "setting slack web hook"
source build/slack_web_hook.sh

if [ -n "$SLACK_WEBHOOK_URL" ]; then
	curl -X POST -H 'Content-type: application/json' \
	--data '{"text":"'":rocket::rocket::rocket:job $SLURM_JOB_ID $SLURM_JOB_NAME starts now on partition $SLURM_JOB_PARTITION"'",
	         "attachments": [{"title": "Argument Details",
	                          "text": "'"running on task \`$*\`\n"'",
	                          "color": "36a64f"}]}' "$SLACK_WEBHOOK_URL"
fi

srun ./run-sims.sh "$@"
ecode="$?"

if [ -n "$SLACK_WEBHOOK_URL" ]; then
	if [ "$ecode" -eq 0 ]; then
		curl -X POST -H 'Content-type: application/json' \
		--data '{"text": "'":checkered_flag::checkered_flag::checkered_flag: job $SLURM_JOB_ID $SLURM_JOB_NAME completed successfully on partition $SLURM_JOB_PARTITION"'",
		         "attachments": [{"title": "Argument Details",
		                          "text": "'"task \`$*\`"'",
		                          "color": "36a64f"}]}' "$SLACK_WEBHOOK_URL"
	else
		curl -X POST -H 'Content-type: application/json' \
		--data '{"text": "'":bomb::bomb::bomb: job $SLURM_JOB_ID $SLURM_JOB_NAME failed on partition $SLURM_JOB_PARTITION"'",
		         "attachments": [{"title": "Argument Details",
		                          "text": "'"task \`$*\` exit with error $ecode \n"'",
		                          "color": "#FF0000"}]}' "$SLACK_WEBHOOK_URL"
	fi
fi
exit "$ecode"

