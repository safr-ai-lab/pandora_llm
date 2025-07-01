# BoW Tokens Random Forest
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['bow_tokens']" --classifier.name "random_forest" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train True --classifier.infer True

# TFIDF Tokens Decision Tree
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['tfidf_tokens']" --classifier.name "decision_tree" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train True --classifier.infer True

# Word2Vec Tokens LogReg
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['word2vec_tokens']" --classifier.name "log_reg" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train True --classifier.infer True

# BERT Tokens NN
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['bert_tokens']" --classifier.name "neural_net" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train True --classifier.infer True

# BoW Text Gradient Boosting
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['bow_text']" --classifier.name "gradient_boosting" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train True --classifier.infer True

# TFIDF Text Word2Vec Text BERT Text Log Reg
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['tfidf_text','word2vec_text','bert_text']" --classifier.name "log_reg" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train True --classifier.infer True

# Loss
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['loss']" --classifier.name "threshold" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train False --classifier.infer True

# LossRatio
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['loss_ratio']" --classifier.name "threshold" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train False --classifier.infer True \
    --loss_ratio.ref_name "EleutherAI/pythia-160m-deduped"

# ZLIB
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['zlib']" --classifier.name "threshold" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train False --classifier.infer True

# LossZLIB
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['loss_zlib']" --classifier.name "threshold" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train False --classifier.infer True

# Mink
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['mink']" --classifier.name "threshold" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train False --classifier.infer True \
    --mink.k 0.1

# Minkxx
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['minkxx']" --classifier.name "threshold" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train False --classifier.infer True \
    --minkxx.k 0.1

# ALoRa
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['alora']" --classifier.name "threshold" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train False --classifier.infer True

# GradNorm
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['gradnorm']" --classifier.name "log_reg" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train True --classifier.infer True \
    --gradnorm.norms "[1,2,'inf']"

# MoPe
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['mope']" --classifier.name "threshold" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train False --classifier.infer True \
    --mope.num_models 2 \
    --mope.noise_stdev 0.01

# ReCaLL
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['recall']" --classifier.name "threshold" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train False --classifier.infer True \
    --recall.prefix_length 100

# DCPDD
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['dcpdd']" --classifier.name "threshold" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train False --classifier.infer True \
    --dcpdd.dataset_name "jsonW0/pile_dedupe_pack_eos_error" \
    --dcpdd.dataset_split "nonmember" \
    --dcpdd.dataset_size 500

# ModelStealing
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['modelstealing']" --classifier.name "log_reg" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train True --classifier.infer True

# JL
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['jl']" --classifier.name "log_reg" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train True --classifier.infer True

# DetectGPT
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['detectgpt']" --classifier.name "threshold" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train False --classifier.infer True \
    --detectgpt.num_perts 3 \

# Quantile
pandora-mia --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 200 --dataset.train_start_index 3000 --dataset.num_val_samples 200 --dataset.val_start_index 0 --features.names "['quantile']" --classifier.name "threshold" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train False --classifier.infer True \
    --quantile.ref_name "EleutherAI/pythia-70m-deduped" \
    --quantile.num_ref 3