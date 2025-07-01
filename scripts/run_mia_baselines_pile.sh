#!/bin/bash
#SBATCH -J baselines         # Name
#SBATCH -t 0-12:00 # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=50000 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o err_out/baselines%j.out # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e err_out/baselines%j.err # File to which STDERR will be written, %j inserts jobid
#SBATCH -p gpu ## specify partition
#SBATCH -n 1
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --account=sneel_lab ## who to charge computer under
#SBATCH --mail-type=END,FAIL #SBATCH --mail-user=jgwang@college.harvard.edu

#!/usr/bin/env bash

# included when I sbatch this 
conda init bash
source activate pandorav2
# pip install pandora_llm
module load cuda

# Datasets
DATASETS=(
  "pythia_dedupe_mia_96975-96990_97010-97025"
  "pythia_dedupe_mia_0-97000_97000-98500"
  "pythia_dedupe_mia_89975-89990_90010-90025"
  "pythia_dedupe_mia_0-90000_90000-98500"
  "pythia_dedupe_mia_89975-89990_90000-98500"
  "pythia_dedupe_mia_0-90000_90010-90025"
  "pythia_dedupe_mia_0-70000_70000-98500"
  "pythia_dedupe_mia_69975-69990_70010-70025"
  "pythia_dedupe_mia_0-50000_50000-98500"
  "pythia_dedupe_mia_49975-49990_50010-50025"
)

MODEL_REVISION=(
  "step97000"
  "step97000"
  "step90000"
  "step90000"
  "step90000"
  "step70000"
  "step70000"
  "step50000"
  "step50000"
)

# First set of features (for log_reg)
FEATURES_LOGREG=(
  "bow_tokens"
  "bow_text"
  "tfidf_text"
  "tfidf_tokens"
  "word2vec_text"
  "word2vec_tokens"
  "bert_text"
  "bert_tokens"
)

# Second set of features (for threshold classifier)
FEATURES_THRESHOLD=(
  "loss"
  "mope"
  "zlib"
  "alora"
  "mink"
)

# Loop over datasets
for i in "${!DATASETS[@]}"; do
  DATASET="${DATASETS[$i]}"
  REVISION="${MODEL_REVISION[$i]}"
  echo "==== Processing dataset: ${DATASET} ===="

  # 2. Now run the single big command that includes all log_reg features at once
  echo "Running log_reg with all features at once..."
  python -m pandora_llm.routines.membership_inference \
    --dataset.name "mfli314/${DATASET}" \
    --dataset.num_train_samples 10000 \
    --dataset.train_start_index 0 \
    --dataset.num_val_samples 2000 \
    --dataset.val_start_index 10000 \
    --features.names "['bow_tokens', 'bow_text', 'tfidf_text', 'tfidf_tokens', 'word2vec_text', 'word2vec_tokens', 'bert_text', 'bert_tokens']" \
    --classifier.name log_reg \
    --model.name "EleutherAI/pythia-1b-deduped" \
    --log_reg.max_iter 500 \
    --features.compute True \
    --classifier.train True \
    --classifier.infer True
  echo

  echo "Running log_reg with all features at once..."
  python -m pandora_llm.routines.membership_inference \
    --dataset.name "mfli314/${DATASET}" \
    --dataset.num_train_samples 10000 \
    --dataset.train_start_index 0 \
    --dataset.num_val_samples 2000 \
    --dataset.val_start_index 10000 \
    --features.names "['bow_tokens', 'bow_text', 'tfidf_text', 'tfidf_tokens', 'word2vec_text', 'word2vec_tokens', 'bert_text', 'bert_tokens']" \
    --classifier.name random_forest \
    --model.name "EleutherAI/pythia-1b-deduped" \
    --log_reg.max_iter 500 \
    --features.compute True \
    --classifier.train True \
    --classifier.infer True
  echo

  # 3. Loop over the second set of features (threshold classifier)
  for FEATURE in "${FEATURES_THRESHOLD[@]}"; do
    echo "Running threshold classifier for feature: ${FEATURE}"
    python -m pandora_llm.routines.membership_inference \
      --dataset.name "mfli314/${DATASET}" \
      --dataset.num_train_samples 10000 \
      --dataset.train_start_index 0 \
      --dataset.num_val_samples 2000 \
      --dataset.val_start_index 10000 \
      --features.names "['${FEATURE}']" \
      --classifier.name threshold \
      --model.name "EleutherAI/pythia-1b-deduped" \
      --features.compute True \
      --classifier.train False \
      --classifier.infer True \
      --model.revision "${REVISION}" 
    echo
  done
done

echo "All experiments completed!"
