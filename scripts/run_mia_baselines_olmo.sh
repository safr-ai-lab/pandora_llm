#!/bin/bash
#SBATCH -J baselines         # Name
#SBATCH -t 0-12:00 # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH --mem=100000 # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o err_out/baselines%j.out # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e err_out/baselines%j.err # File to which STDERR will be written, %j inserts jobid
#SBATCH --partition gpu,seas_gpu,gpu_requeue  
#SBATCH -n 1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --account=sitanc_lab  ## who to charge computer under
#SBATCH --mail-type=END,FAIL #SBATCH --mail-user=marvinli@college.harvard.edu

#!/usr/bin/env bash

# included when I sbatch this 
conda init bash
# pip install pandora_llm
module load cuda

# Datasets
DATASETS=(
  "OLMo-7B-mia-n10000"
)

MODEL_REVISION=(
  "step400000-tokens1769B"
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
  # 1. Run logreg with each feature individually 
  for FEATURE in "${FEATURES_LOGREG[@]}"; do
    echo "Running log_reg with feature: ${FEATURE}"
    python -m pandora_llm.routines.membership_inference \
      --dataset.name "mfli314/${DATASET}" \
      --dataset.num_train_samples 8000 \
      --dataset.train_start_index 0 \
      --dataset.num_val_samples 2000 \
      --dataset.val_start_index 8000 \
      --features.names "['${FEATURE}']" \
      --classifier.name log_reg \
      --model.name "allenai/OLMo-7B" \
      --log_reg.max_iter 500 \
      --features.compute True \
      --classifier.train True \
      --classifier.infer True \
      --model.revision "${REVISION}" 
    echo
  done  
  
  # 2. Now run the single big command that includes all log_reg features at once
  echo "Running log_reg with all features at once..."
  python -m pandora_llm.routines.membership_inference \
    --dataset.name "mfli314/${DATASET}" \
    --dataset.num_train_samples 8000 \
    --dataset.train_start_index 0 \
    --dataset.num_val_samples 2000 \
    --dataset.val_start_index 8000 \
    --features.names "['bow_tokens', 'bow_text', 'tfidf_text', 'tfidf_tokens', 'word2vec_text', 'word2vec_tokens', 'bert_text', 'bert_tokens']" \
    --classifier.name log_reg \
    --model.name "allenai/OLMo-7B" \
    --log_reg.max_iter 500 \
    --features.compute True \
    --classifier.train True \
    --classifier.infer True \
    --model.revision "${REVISION}" 
  echo

  echo "Running log_reg with all features at once..."
  python -m pandora_llm.routines.membership_inference \
    --dataset.name "mfli314/${DATASET}" \
    --dataset.num_train_samples 8000 \
    --dataset.train_start_index 0 \
    --dataset.num_val_samples 2000 \
    --dataset.val_start_index 8000 \
    --features.names "['bow_tokens', 'bow_text', 'tfidf_text', 'tfidf_tokens', 'word2vec_text', 'word2vec_tokens', 'bert_text', 'bert_tokens']" \
    --classifier.name random_forest \
    --model.name "allenai/OLMo-7B" \
    --log_reg.max_iter 500 \
    --features.compute True \
    --classifier.train True \
    --classifier.infer True \
    --model.revision "${REVISION}" 
  echo

  # 3. Loop over the second set of features (threshold classifier)
  for FEATURE in "${FEATURES_THRESHOLD[@]}"; do
    echo "Running threshold classifier for feature: ${FEATURE}"
    python -m pandora_llm.routines.membership_inference \
      --dataset.name "mfli314/${DATASET}" \
      --dataset.num_train_samples 8000 \
      --dataset.train_start_index 0 \
      --dataset.num_val_samples 2000 \
      --dataset.val_start_index 8000 \
      --features.names "['${FEATURE}']" \
      --classifier.name threshold \
      --model.name "allenai/OLMo-7B" \
      --features.compute True \
      --classifier.train False \
      --classifier.infer True \
      --model.revision "${REVISION}" 
    echo
  done
done

echo "All experiments completed!"
