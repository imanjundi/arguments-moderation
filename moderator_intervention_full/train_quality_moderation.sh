#!/bin/bash

args=(
  --moderation_type quality_moderation
  --in_domain True
  --project_name "${SWEEP_PROJECT:-regulationroom-moderator-intervention-improve-quality}"
  )
bash moderator_intervention_full/train_base.sh "${args[@]}" "$@"