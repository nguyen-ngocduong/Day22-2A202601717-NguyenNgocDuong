#!/bin/bash
set -e
pref-lab validate data/sample_preferences.jsonl
pref-lab evaluate --config configs/local.yaml
cat outputs/metrics.json
