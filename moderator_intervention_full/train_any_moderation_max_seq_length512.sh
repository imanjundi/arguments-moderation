#!/bin/bash

args=(
  # data arguments
  --max_seq_length 512
  # training arguments
  --gradient_accumulation_steps 2
  --per_device_train_batch_size 8
  )
bash moderator_intervention_full/train_any_moderation.sh "${args[@]}" "$@"