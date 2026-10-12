#!/usr/bin/env bash

set -e

world_size=4
device_memory=45
num_microbatches=16
batch_size=256

model=gpt2_xl
model_config=config/${model}.yaml

mbs=$((batch_size / num_microbatches))
profile_dict=profile_dicts/${model}_${mbs}.json

if [ ! -e "$profile_dict" ]; then
	echo "Profile '$model' with microbatch size $mbs."
	python -m src.profiling_gpt2 \
		--num-microbatches=$num_microbatches \
		--batch-size=$batch_size \
		--optimizer-config="config/optimizer.yaml" \
		--model-config=$model_config \
		--output-file=$profile_dict
	echo "Profiling finished."
fi


spec_file=config/dp_${model}_${wolrld_size}pp_${batch_size}gbs_${num_microbatches}nmb_${device_memory}mem.yaml

if [ ! -e "$spec_file" ]; then
	PYTHONPATH=. python algorithms/dp.py \
		--world-size=$world_size \
		--device-memory=$device_memory \
		--num-microbatches=$num_microbatches \
		--batch-size=$batch_size \
		--model-config=$model_config \
		--profiled-data=$profile_dict \
		--output-file=$spec_file
fi

torchrun --nproc_per_node=$world_size -m src.train_gpt2 \
		--num-microbatches=$num_microbatches \
		--batch-size=$batch_size \
		--optimizer-config="config/optimizer.yaml" \
		--model-config=$model_config \
		--spec=$spec_file \
		--setup=dp_${model}_${batch_size}gbs_${num_microbatches}nmb_${device_memory}mem \
