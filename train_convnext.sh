#!/usr/bin/env bash

world_size=4
device_memory=43
num_microbatches=16
batch_size=256

model=convnext_base
expansion_pattern=32,32,32,128,128,128,64,64,64
expansion_tag=${expansion_pattern//,/-}

mbs=$((batch_size / num_microbatches))
profile_dict=profile_dicts/${model}_exp${expansion_tag}_${mbs}.json

if [ ! -e "$profile_dict" ]; then
	echo "Profile '$model' with microbatch size $mbs."
	python -m src.profiling_convnext \
		--num-microbatches=$num_microbatches \
		--batch-size=$batch_size \
		--optimizer-config="config/optimizer.yaml" \
		--model=$model \
		--expansion-pattern=$expansion_pattern \
		--output-file=$profile_dict
	echo "Profiling finished."
fi

spec_file=config/dp_${model}_exp${expansion_tag}_${world_size}pp_${batch_size}gbs_${num_microbatches}nmb_${device_memory}mem.yaml

if [ ! -e "$spec_file" ]; then
	PYTHONPATH=. python algorithms/dp_convnext.py \
		--world-size=$world_size \
		--device-memory=$device_memory \
		--num-microbatches=$num_microbatches \
		--batch-size=$batch_size \
		--model=$model \
		--profiled-data=$profile_dict \
		--output-file=$spec_file
fi

torchrun --nproc_per_node=$world_size -m src.train_convnext \
		--num-microbatches=$num_microbatches \
		--batch-size=$batch_size \
			--optimizer-config="config/optimizer.yaml" \
			--model=$model \
			--expansion-pattern=$expansion_pattern \
			--spec=$spec_file \
			--setup=adapipe_${model}_exp${expansion_tag}_${batch_size}gbs_${num_microbatches}nmb_${device_memory}mem
