#!/usr/bin/env bash

world_size=4
device_memory=39
num_microbatches=16
batch_size=512

model=resnet152x8
profile_model=resnetx8

mbs=$((batch_size / num_microbatches))
profile_dict=profile_dicts/${profile_model}_${mbs}.json

if [ ! -e "$profile_dict" ]; then
	echo "Profile '$model' with microbatch size $mbs."
	python -m src.profiling_rn \
		--num-microbatches=$num_microbatches \
		--batch-size=$batch_size \
		--optimizer-config="config/optimizer.yaml" \
		--model=$profile_model \
		--output-file=$profile_dict
	echo "Profiling finished."
fi

spec_file=config/dp_${model}_${wolrld_size}pp_${batch_size}gbs_${num_microbatches}nmb_${device_memory}mem.yaml

if [ ! -e "$spec_file" ]; then
	PYTHONPATH=. python algorithms/dp_rn.py \
		--world-size=$world_size \
		--device-memory=$device_memory \
		--num-microbatches=$num_microbatches \
		--batch-size=$batch_size \
		--model=$model \
		--profiled-data=$profile_dict \
		--output-file=$spec_file
fi

torchrun --nproc_per_node=$world_size -m src.train_rn \
		--num-microbatches=$num_microbatches \
		--batch-size=$batch_size \
		--optimizer-config="config/optimizer.yaml" \
		--model=$model \
		--spec=$spec_file \
		--setup=dp_${model}_${batch_size}gbs_${num_microbatches}nmb_${device_memory}mem
