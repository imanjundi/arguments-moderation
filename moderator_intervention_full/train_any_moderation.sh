#!/bin/bash

args=(
  --moderation_type any_moderation
  --project_name "${SWEEP_PROJECT:-regulationroom-moderator-intervention-full}"
  )
bash moderator_intervention_full/train_base.sh "${args[@]}" "$@"