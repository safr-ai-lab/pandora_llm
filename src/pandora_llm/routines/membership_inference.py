import os
import time
import jsonargparse
import wandb
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
from pandora_llm.data import load_dataset_with_metadata, ThePile
from pandora_llm.features import FeatureSet, BoW, TFIDF, Word2Vec, BertFeatureComputer, LOSS, LossRatio, ZLIB, LossZLIB, MinK, MinKXX, ALoRa, GradNorm, MoPe, ReCaLL, DCPDD, ModelStealing, JL, DetectGPT, Quantile
from pandora_llm.classifiers import Threshold, DecisionTree, RandomForest, GradientBoosting, LogReg, NeuralNet
from pandora_llm.utils.plot_utils import plot_ROC, plot_histogram
from pandora_llm.utils.log_utils import get_my_logger, get_git_hash, clean_filename, mem_stats
device = "cuda" if torch.cuda.is_available() else "cpu"

"""
python -m pandora_llm.routines.membership_inference --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 1000 --dataset.train_start_index 3000 --dataset.num_val_samples 1000 --dataset.val_start_index 0 --features.names "['bow_tokens']" --classifier.name "random_forest" --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train True --classifier.infer True
python -m pandora_llm.routines.membership_inference --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 0 --dataset.train_start_index 3000 --dataset.num_val_samples 1000 --dataset.val_start_index 0 --features.names "['loss']" --classifier.name threshold --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train False --classifier.infer True 
python -m pandora_llm.routines.membership_inference --dataset.name "jsonW0/pile_dedupe_pack_eos_error" --dataset.num_train_samples 0 --dataset.train_start_index 3000 --dataset.num_val_samples 1000 --dataset.val_start_index 0 --features.names "['divergence']" --classifier.name threshold --model.name "EleutherAI/pythia-70m-deduped" --features.compute True --classifier.train False --classifier.infer True 
"""

def add_dataset_arguments(parser: jsonargparse.ArgumentParser):
    parser.add_argument("--dataset.name", type=str, required=False, help="Dataset name")
    parser.add_argument("--dataset.num_train_samples", type=int, required=False, default=0, help="Number of train samples to take from the dataset - set to 0 for unsupervised MIA")
    parser.add_argument("--dataset.num_val_samples", type=int, required=False, default=0, help="Number of val samples to take from the dataset - set to 0 for unsupervised MIA")
    parser.add_argument("--dataset.train_start_index", type=int, required=False, default=0, help="Slice dataset starting from this index")
    parser.add_argument("--dataset.val_start_index", type=int, required=False, default=0, help="Slice dataset starting from this index")
    parser.add_argument("--dataset.batch_size", type=int, required=False, default=1, help="Batch size")

def add_model_arguments(parser: jsonargparse.ArgumentParser):
    parser.add_argument("--model.name", type=str, required=False, help="Huggingface model name")
    parser.add_argument("--model.revision", type=str, required=False, help="Huggingface model revision")
    parser.add_argument("--model.cache_dir", type=str, required=False, help="Huggingface model cache directory")

def add_features_arguments(parser: jsonargparse.ArgumentParser):
    parser.add_argument("--features.compute", type=bool, required=False, default=True, help="Whether to compute features")
    parser.add_argument("--features.member_train_paths", type=list[str], required=False, default=[], help="Member features paths (train)")
    parser.add_argument("--features.member_val_paths", type=list[str], required=False, default=[], help="Member features paths (val)")
    parser.add_argument("--features.nonmember_train_paths", type=list[str], required=False, default=[], help="Nonmember features paths (train)")
    parser.add_argument("--features.nonmember_val_paths", type=list[str], required=False, default=[], help="Nonmember features paths (val)")
    parser.add_argument("--features.names", type=list[str], required=False, default=[], help="List of features to use")
    
    parser.add_argument("--loss_ratio.ref_name", type=str, required=False, help="Reference model name")
    parser.add_argument("--loss_ratio.ref_revision", type=str, required=False, help="Reference model revision")
    parser.add_argument("--loss_ratio.ref_cache_dir", type=str, required=False, help="Refrence model cache dir")
    parser.add_argument("--mink.k", type=float, required=False, default=0.1, help="Compute loss on top-k least likely tokens")
    parser.add_argument("--minkxx.k", type=float, required=False, default=0.1, help="Compute loss on top-k least likely (z-scored) tokens")
    parser.add_argument("--alora.learning_rate", type=float, required=False, default=5e-5, help="Learning rate for ALoRa")
    parser.add_argument("--mope.num_models", type=int, required=False, help="Number of new models")
    parser.add_argument("--mope.noise_stdev", type=float, required=False, help="Noise standard deviation")
    parser.add_argument("--mope.noise_type", type=str, required=False, default="gaussian", help="Noise to add to model. Either `gaussian` or `rademacher`")
    parser.add_argument("--detectgpt.num_perts", type=int, required=False, help="Number of perturbations to apply")
    parser.add_argument("--gradnorm.norms", type=list, required=False, default=[1,2,"inf"], help="Norms to use. Use the string `inf` to specify inf-norm.")
    parser.add_argument("--modelstealing.project_type", type=str, required=False, default="rademacher", help="type of projection (rademacher or normal)")
    parser.add_argument("--modelstealing.proj_seed", type=int, required=False, default=229, help="Seed for random projection")
    parser.add_argument("--modelstealing.proj_dim_last", type=int, required=False, default=512, help="Dimension of projection of last layer gradients for model stealing. Default = 512.")
    parser.add_argument("--jl.mode", type=str, required=False, default="layerwise", help="JL compute mode")
    parser.add_argument("--jl.proj_type", type=str, required=False, default="rademacher", help="type of projection (rademacher or normal)")
    parser.add_argument("--jl.proj_seed", type=int, required=False, default=229, help="Seed for random projection")
    parser.add_argument("--jl.proj_dim_x", type=int, required=False, default=32, help="Project grad wrt x to this dim. Default = 32.")
    parser.add_argument("--jl.proj_dim_layer", type=int, required=False, default=3, help="When JLing a layer, project to this dimension. Default = 3.")
    parser.add_argument("--jl.proj_dim_group", type=int, required=False, default=512, help="When JLing a group, project to this dimension. Default = 512.")
    parser.add_argument("--jl.num_splits", type=int, required=False, default=8, help="Num groups of layers to try dividing into before JL projection. Default = 8.")
    parser.add_argument("--recall.prefix", type=str, required=False, help="Prefix for ReCaLL")
    parser.add_argument("--recall.prefix_length", type=int, required=False, help="Length of prefix for ReCaLL")
    parser.add_argument("--dcpdd.dataset_name", type=str, required=False, help="Dataset name to compute reference vocab frequency")
    parser.add_argument("--dcpdd.dataset_subset", type=str, required=False, help="Dataset subset to compute reference vocab frequency")
    parser.add_argument("--dcpdd.dataset_split", type=str, required=False, help="Dataset split to compute reference vocab frequency")
    parser.add_argument("--dcpdd.dataset_size", type=int, required=False, help="Size of reference dataset to compute reference vocab frequency")
    parser.add_argument("--quantile.ref_name", type=str, required=False, help="Reference model name")
    parser.add_argument("--quantile.ref_revision", type=str, required=False, help="Reference model revision")
    parser.add_argument("--quantile.ref_cache_dir", type=str, required=False, help="Refrence model cache dir")
    parser.add_argument("--quantile.num_ref", type=int, required=False, default=4, help="Number of reference models in ensemble")

def add_classifier_arguments(parser: jsonargparse.ArgumentParser):
    parser.add_argument("--classifier.train", type=bool, required=False, default=True, help="Whether to train classifier")
    parser.add_argument("--classifier.infer", type=bool, required=False, default=True, help="Whether to infer membership")
    parser.add_argument("--classifier.name", type=str, required=False, help="The classifier to use")
    parser.add_argument("--classifier.path", type=str, required=False, help="The path to load/save classifier")

    parser.add_argument("--decision_tree.max_depth", type=int, required=False, help="Max depth for decision tree fitting")
    parser.add_argument("--random_forest.n_estimators", type=int, required=False, default=100, help="Number of estimators for random forest fitting")
    parser.add_argument("--random_forest.max_depth", type=int, required=False, help="Max depth for random forest fitting")
    parser.add_argument("--gradient_boosting.max_iter", type=int, required=False, default=100, help="Max number of iterations for gradient boosting fitting")
    parser.add_argument("--gradient_boosting.max_depth", type=int, required=False, help="Max depth for gradient boosting fitting")
    parser.add_argument("--log_reg.max_iter", type=int, required=False, default=1000, help="Max number of iterations for logistic regression fitting")
    parser.add_argument("--neural_net.size", type=str, required=False, default="small", help="Size of neural network")
    parser.add_argument("--neural_net.num_epochs", type=int, required=False, default=10, help="Number of epochs to train neural network")
    parser.add_argument("--neural_net.batch_size", type=int, required=False, default=128, help="Batch size to train neural network")

def add_device_arguments(parser: jsonargparse.ArgumentParser):
    parser.add_argument("--device.seed", type=int, required=False, default=229, help="Random seed")

def add_wandb_arguments(parser: jsonargparse.ArgumentParser):
    parser.add_argument("--wandb.enable", type=bool, required=False, default=False, help="Use wandb")
    parser.add_argument("--wandb.project", type=str, required=False, default="pandora", help="Wandb project name")
    parser.add_argument("--wandb.team", type=str, required=False, default=None, help="Wandb team name")
    parser.add_argument("--wandb.git_hash", type=str, required=False, help="Git hash; will be overwritten if in git repo")

def main():
    start_total = time.perf_counter()
    ####################################################################################################
    # 0. SETUP
    ####################################################################################################
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--config", action="config")  
    parser.add_argument("--experiment_name", type=str, required=False, help="Experiment name")
    parser.add_argument("--tag", type=str, required=False, help="If using default name, add more information of your choice")
    add_device_arguments(parser)
    add_wandb_arguments(parser)
    add_dataset_arguments(parser)
    add_model_arguments(parser)
    add_features_arguments(parser)
    add_classifier_arguments(parser)
    args = parser.parse_args()

    accelerator = Accelerator()
    set_seed(args.device.seed)
    
    exp_name_base = "experiment"
    if args.experiment_name is None:
        ftssep = ",".join(args.features.names)
        exp_name_base = clean_filename(
            (f"F={ftssep}__C={args.classifier.name}") + 
            (f"__M={args.model.name}" if args.model.name is not None else "") + (f"__{args.model.revision}" if args.model.revision is not None else "") +
            (f"__D={args.dataset.name}__N={args.dataset.num_train_samples},{args.dataset.num_val_samples}") + 
            (f"__tag={args.tag}" if args.tag is not None else "")
        )
        print(f"Experiment_Name: {exp_name_base}")
        classifier_exp_name = f"results-{clean_filename(ftssep)}"
        args.experiment_name = os.path.join("results", exp_name_base, exp_name_base)
        print(f"Name: {args.experiment_name}")

    args.model.cache_dir = args.model.cache_dir if (args.model.cache_dir is not None or args.model.name is None) else os.path.join("models",clean_filename(args.model.name))
    args.classifier.path = args.classifier.path if args.classifier.path is not None else os.path.join("classifiers",args.classifier.name,exp_name_base)+".pt"
    print(f"Classifier path: {args.classifier.path}")
    os.makedirs(os.path.dirname(args.experiment_name), exist_ok=True)
    logger = get_my_logger(log_file=f"{args.experiment_name}.log")
    args.wandb.git_hash = get_git_hash() if get_git_hash() else args.git_hash
    parser.save(args,f"{args.experiment_name}_args.yaml", overwrite=True)
    if args.wandb.enable:
        wandb.init(
            config=args,
            project=args.wandb.project,
            entity=args.wandb.team,
            name=args.experiment_name.split("/")[-1],
        )
    ####################################################################################################
    # 1. COMPUTE FEATURES
    ####################################################################################################
    if args.features.compute:
        logger.info("Loading Data")
        start = time.perf_counter()
    
        try:
            dataset = load_dataset_with_metadata(args.dataset.name, trust_remote_code=True)
        except:
            raise ValueError(f"Unrecognized dataset: {args.dataset.name}")

        config = AutoConfig.from_pretrained(args.model.name, revision=args.model.revision, cache_dir=args.model.cache_dir,trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model.name, revision=args.model.revision, cache_dir=args.model.cache_dir,trust_remote_code=True)

        member_train_dataset = dataset["member"].select(range(args.dataset.train_start_index,args.dataset.train_start_index+args.dataset.num_train_samples))
        member_val_dataset = dataset["member"].select(range(args.dataset.val_start_index,args.dataset.val_start_index+args.dataset.num_val_samples))
        nonmember_train_dataset = dataset["nonmember"].select(range(args.dataset.train_start_index,args.dataset.train_start_index+args.dataset.num_train_samples))
        nonmember_val_dataset = dataset["nonmember"].select(range(args.dataset.val_start_index,args.dataset.val_start_index+args.dataset.num_val_samples))
        
        member_train_text_dl = DataLoader(member_train_dataset["text"], batch_size=args.dataset.batch_size)
        member_val_text_dl = DataLoader(member_val_dataset["text"], batch_size=args.dataset.batch_size)
        nonmember_train_text_dl = DataLoader(nonmember_train_dataset["text"], batch_size=args.dataset.batch_size)
        nonmember_val_text_dl = DataLoader(nonmember_val_dataset["text"], batch_size=args.dataset.batch_size)

        member_train_tokens_dl = DataLoader(member_train_dataset["tokens"], batch_size=args.dataset.batch_size, collate_fn=lambda batch: torch.tensor(batch, dtype=torch.long))
        member_val_tokens_dl = DataLoader(member_val_dataset["tokens"], batch_size=args.dataset.batch_size, collate_fn=lambda batch: torch.tensor(batch, dtype=torch.long))
        nonmember_train_tokens_dl = DataLoader(nonmember_train_dataset["tokens"], batch_size=args.dataset.batch_size, collate_fn=lambda batch: torch.tensor(batch, dtype=torch.long))
        nonmember_val_tokens_dl = DataLoader(nonmember_val_dataset["tokens"], batch_size=args.dataset.batch_size, collate_fn=lambda batch: torch.tensor(batch, dtype=torch.long))
        end = time.perf_counter()
        logger.info(f"- Dataset loading took {end-start} seconds.")
        ####################################################################################################
        logger.info("Computing Features")
        start = time.perf_counter()

        member_train_features = FeatureSet()
        member_val_features = FeatureSet()
        nonmember_train_features = FeatureSet()
        nonmember_val_features = FeatureSet()

        if "bow_text" in args.features.names:
            feature_computer = BoW()
            feature_computer.train_bow_text(DataLoader(member_train_dataset["text"]+nonmember_train_dataset["text"], batch_size=args.dataset.batch_size))
            member_train_features["bow_text"] = feature_computer.compute_features(member_train_text_dl, mode="text")
            member_val_features["bow_text"] = feature_computer.compute_features(member_val_text_dl, mode="text")
            nonmember_train_features["bow_text"] = feature_computer.compute_features(nonmember_train_text_dl, mode="text")
            nonmember_val_features["bow_text"] = feature_computer.compute_features(nonmember_val_text_dl, mode="text")
        if "bow_tokens" in args.features.names: 
            feature_computer = BoW()
            feature_computer.train_bow_tokens(DataLoader(member_train_dataset["tokens"]+nonmember_train_dataset["tokens"], batch_size=args.dataset.batch_size))
            member_train_features["bow_tokens"] = feature_computer.compute_features(member_train_tokens_dl, mode="tokens")
            member_val_features["bow_tokens"] = feature_computer.compute_features(member_val_tokens_dl, mode="tokens")
            nonmember_train_features["bow_tokens"] = feature_computer.compute_features(nonmember_train_tokens_dl, mode="tokens")
            nonmember_val_features["bow_tokens"] = feature_computer.compute_features(nonmember_val_tokens_dl, mode="tokens")
        if "tfidf_text" in args.features.names:
            feature_computer = TFIDF()
            feature_computer.train_tfidf_text(DataLoader(member_train_dataset["text"]+nonmember_train_dataset["text"], batch_size=args.dataset.batch_size))
            member_train_features["tfidf_text"] = feature_computer.compute_features(member_train_text_dl, mode="text")
            member_val_features["tfidf_text"] = feature_computer.compute_features(member_val_text_dl, mode="text")
            nonmember_train_features["tfidf_text"] = feature_computer.compute_features(nonmember_train_text_dl, mode="text")
            nonmember_val_features["tfidf_text"] = feature_computer.compute_features(nonmember_val_text_dl, mode="text")
        if "tfidf_tokens" in args.features.names: 
            feature_computer = TFIDF()
            feature_computer.train_tfidf_tokens(DataLoader(member_train_dataset["tokens"]+nonmember_train_dataset["tokens"], batch_size=args.dataset.batch_size))
            member_train_features["tfidf_tokens"] = feature_computer.compute_features(member_train_tokens_dl, mode="tokens")
            member_val_features["tfidf_tokens"] = feature_computer.compute_features(member_val_tokens_dl, mode="tokens")
            nonmember_train_features["tfidf_tokens"] = feature_computer.compute_features(nonmember_train_tokens_dl, mode="tokens")
            nonmember_val_features["tfidf_tokens"] = feature_computer.compute_features(nonmember_val_tokens_dl, mode="tokens")
        if "word2vec_text" in args.features.names:
            feature_computer = Word2Vec()
            feature_computer.load_pretrained_model()
            member_train_features["word2vec_text"] = feature_computer.compute_features(member_train_text_dl, mode="text")
            member_val_features["word2vec_text"] = feature_computer.compute_features(member_val_text_dl, mode="text")
            nonmember_train_features["word2vec_text"] = feature_computer.compute_features(nonmember_train_text_dl, mode="text")
            nonmember_val_features["word2vec_text"] = feature_computer.compute_features(nonmember_val_text_dl, mode="text")
        if "word2vec_tokens" in args.features.names: 
            feature_computer = Word2Vec()
            feature_computer.load_pretrained_model()
            member_train_features["word2vec_tokens"] = feature_computer.compute_features(member_train_tokens_dl, mode="tokens", tokenizer=tokenizer)
            member_val_features["word2vec_tokens"] = feature_computer.compute_features(member_val_tokens_dl, mode="tokens", tokenizer=tokenizer)
            nonmember_train_features["word2vec_tokens"] = feature_computer.compute_features(nonmember_train_tokens_dl, mode="tokens", tokenizer=tokenizer)
            nonmember_val_features["word2vec_tokens"] = feature_computer.compute_features(nonmember_val_tokens_dl, mode="tokens", tokenizer=tokenizer)
        if "bert_text" in args.features.names:
            feature_computer = BertFeatureComputer()
            feature_computer.load_pretrained_model()
            member_train_features["bert_text"] = feature_computer.compute_features(member_train_text_dl, mode="text", accelerator=accelerator)
            member_val_features["bert_text"] = feature_computer.compute_features(member_val_text_dl, mode="text", accelerator=accelerator)
            nonmember_train_features["bert_text"] = feature_computer.compute_features(nonmember_train_text_dl, mode="text", accelerator=accelerator)
            nonmember_val_features["bert_text"] = feature_computer.compute_features(nonmember_val_text_dl, mode="text", accelerator=accelerator)
            feature_computer.unload_model()
        if "bert_tokens" in args.features.names:
            feature_computer = BertFeatureComputer()
            feature_computer.load_pretrained_model()
            member_train_features["bert_tokens"] = feature_computer.compute_features(member_train_tokens_dl, mode="tokens", accelerator=accelerator, tokenizer=tokenizer)
            member_val_features["bert_tokens"] = feature_computer.compute_features(member_val_tokens_dl, mode="tokens", accelerator=accelerator, tokenizer=tokenizer)
            nonmember_train_features["bert_tokens"] = feature_computer.compute_features(nonmember_train_tokens_dl, mode="tokens", accelerator=accelerator, tokenizer=tokenizer)
            nonmember_val_features["bert_tokens"] = feature_computer.compute_features(nonmember_val_tokens_dl, mode="tokens", accelerator=accelerator, tokenizer=tokenizer)
            feature_computer.unload_model()
        if "loss" in args.features.names:
            feature_computer = LOSS(model_name=args.model.name,model_revision=args.model.revision,model_cache_dir=args.model.cache_dir)
            feature_computer.load_model()
            member_train_features["loss"] = feature_computer.compute_features(member_train_tokens_dl,accelerator=accelerator)
            member_val_features["loss"] = feature_computer.compute_features(member_val_tokens_dl,accelerator=accelerator)
            nonmember_train_features["loss"] = feature_computer.compute_features(nonmember_train_tokens_dl,accelerator=accelerator)
            nonmember_val_features["loss"] = feature_computer.compute_features(nonmember_val_tokens_dl,accelerator=accelerator)
            feature_computer.unload_model()
        if "loss_ratio" in args.features.names:
            feature_computer = LossRatio(model_name=args.model.name,model_revision=args.model.revision,model_cache_dir=args.model.cache_dir,
                                         ref_model_name=args.loss_ratio.ref_name,ref_model_revision=args.loss_ratio.ref_revision,ref_model_cache_dir=args.loss_ratio.ref_cache_dir)
            feature_computer.load_model("primary")
            member_train_features["loss_ratio_primary"] = feature_computer.compute_features(member_train_tokens_dl,accelerator=accelerator)
            member_val_features["loss_ratio_primary"] = feature_computer.compute_features(member_val_tokens_dl,accelerator=accelerator)
            nonmember_train_features["loss_ratio_primary"] = feature_computer.compute_features(nonmember_train_tokens_dl,accelerator=accelerator)
            nonmember_val_features["loss_ratio_primary"] = feature_computer.compute_features(nonmember_val_tokens_dl,accelerator=accelerator)
            feature_computer.unload_model()
            feature_computer.load_model("ref")
            member_train_features["loss_ratio_ref"] = feature_computer.compute_features(member_train_tokens_dl,accelerator=accelerator)
            member_val_features["loss_ratio_ref"] = feature_computer.compute_features(member_val_tokens_dl,accelerator=accelerator)
            nonmember_train_features["loss_ratio_ref"] = feature_computer.compute_features(nonmember_train_tokens_dl,accelerator=accelerator)
            nonmember_val_features["loss_ratio_ref"] = feature_computer.compute_features(nonmember_val_tokens_dl,accelerator=accelerator)
            feature_computer.unload_model()
            member_train_features["loss_ratio"] = LossRatio.reduce(member_train_features["loss_ratio_primary"],member_train_features["loss_ratio_ref"])
            member_val_features["loss_ratio"] = LossRatio.reduce(member_val_features["loss_ratio_primary"],member_val_features["loss_ratio_ref"])
            nonmember_train_features["loss_ratio"] = LossRatio.reduce(nonmember_train_features["loss_ratio_primary"],nonmember_train_features["loss_ratio_ref"])
            nonmember_val_features["loss_ratio"] = LossRatio.reduce(nonmember_val_features["loss_ratio_primary"],nonmember_val_features["loss_ratio_ref"])
        if "zlib" in args.features.names:
            feature_computer = ZLIB()
            member_train_features["zlib"] = feature_computer.compute_features(member_train_text_dl)
            member_val_features["zlib"] = feature_computer.compute_features(member_val_text_dl)
            nonmember_train_features["zlib"] = feature_computer.compute_features(nonmember_train_text_dl)
            nonmember_val_features["zlib"] = feature_computer.compute_features(nonmember_val_text_dl)
        if "loss_zlib" in args.features.names:
            feature_computer = LossZLIB(model_name=args.model.name,model_revision=args.model.revision,model_cache_dir=args.model.cache_dir)
            feature_computer.load_model()
            member_train_features["loss_zlib_loss"] = feature_computer.compute_features(member_train_tokens_dl,accelerator=accelerator,mode="primary")
            member_val_features["loss_zlib_loss"] = feature_computer.compute_features(member_val_tokens_dl,accelerator=accelerator,mode="primary")
            nonmember_train_features["loss_zlib_loss"] = feature_computer.compute_features(nonmember_train_tokens_dl,accelerator=accelerator,mode="primary")
            nonmember_val_features["loss_zlib_loss"] = feature_computer.compute_features(nonmember_val_tokens_dl,accelerator=accelerator,mode="primary")
            feature_computer.unload_model()
            member_train_features["loss_zlib_zlib"] = feature_computer.compute_features(member_train_text_dl,mode="ref")
            member_val_features["loss_zlib_zlib"] = feature_computer.compute_features(member_val_text_dl,mode="ref")
            nonmember_train_features["loss_zlib_zlib"] = feature_computer.compute_features(nonmember_train_text_dl,mode="ref")
            nonmember_val_features["loss_zlib_zlib"] = feature_computer.compute_features(nonmember_val_text_dl,mode="ref")
            member_train_features["loss_zlib"] = LossZLIB.reduce(member_train_features["loss_zlib_loss"],member_train_features["loss_zlib_zlib"])
            member_val_features["loss_zlib"] = LossZLIB.reduce(member_val_features["loss_zlib_loss"],member_val_features["loss_zlib_zlib"])
            nonmember_train_features["loss_zlib"] = LossZLIB.reduce(nonmember_train_features["loss_zlib_loss"],nonmember_train_features["loss_zlib_zlib"])
            nonmember_val_features["loss_zlib"] = LossZLIB.reduce(nonmember_val_features["loss_zlib_loss"],nonmember_val_features["loss_zlib_zlib"])
        if "mink" in args.features.names:
            feature_computer = MinK(model_name=args.model.name,model_revision=args.model.revision,model_cache_dir=args.model.cache_dir)
            feature_computer.load_model()
            member_train_features["token_loss"] = feature_computer.compute_features(member_train_tokens_dl,accelerator=accelerator)
            member_val_features["token_loss"] = feature_computer.compute_features(member_val_tokens_dl,accelerator=accelerator)
            nonmember_train_features["token_loss"] = feature_computer.compute_features(nonmember_train_tokens_dl,accelerator=accelerator)
            nonmember_val_features["token_loss"] = feature_computer.compute_features(nonmember_val_tokens_dl,accelerator=accelerator)
            member_train_features["mink"] = MinK.reduce(member_train_features["token_loss"],k=args.mink.k)
            member_val_features["mink"] = MinK.reduce(member_val_features["token_loss"],k=args.mink.k)
            nonmember_train_features["mink"] = MinK.reduce(nonmember_train_features["token_loss"],k=args.mink.k)
            nonmember_val_features["mink"] = MinK.reduce(nonmember_val_features["token_loss"],k=args.mink.k)
            feature_computer.unload_model()
        if "minkxx" in args.features.names:
            feature_computer = MinKXX(model_name=args.model.name,model_revision=args.model.revision,model_cache_dir=args.model.cache_dir)
            feature_computer.load_model()
            member_train_features["minkxx_z"] = feature_computer.compute_features(member_train_tokens_dl,accelerator=accelerator)
            member_val_features["minkxx_z"] = feature_computer.compute_features(member_val_tokens_dl,accelerator=accelerator)
            nonmember_train_features["minkxx_z"] = feature_computer.compute_features(nonmember_train_tokens_dl,accelerator=accelerator)
            nonmember_val_features["minkxx_z"] = feature_computer.compute_features(nonmember_val_tokens_dl,accelerator=accelerator)
            member_train_features["minkxx"] = MinKXX.reduce(member_train_features["minkxx_z"],k=args.mink.k)
            member_val_features["minkxx"] = MinKXX.reduce(member_val_features["minkxx_z"],k=args.mink.k)
            nonmember_train_features["minkxx"] = MinKXX.reduce(nonmember_train_features["minkxx_z"],k=args.mink.k)
            nonmember_val_features["minkxx"] = MinKXX.reduce(nonmember_val_features["minkxx_z"],k=args.mink.k)
            feature_computer.unload_model()
        if "alora" in args.features.names:
            feature_computer = ALoRa(model_name=args.model.name,model_revision=args.model.revision,model_cache_dir=args.model.cache_dir)
            feature_computer.load_model()
            member_train_features["alora_stepped"], member_train_features["alora_base"] = feature_computer.compute_features(member_train_tokens_dl,learning_rate=args.alora.learning_rate,accelerator=accelerator)
            member_val_features["alora_stepped"], member_val_features["alora_base"] = feature_computer.compute_features(member_val_tokens_dl,learning_rate=args.alora.learning_rate,accelerator=accelerator)
            nonmember_train_features["alora_stepped"], nonmember_train_features["alora_base"] = feature_computer.compute_features(nonmember_train_tokens_dl,learning_rate=args.alora.learning_rate,accelerator=accelerator)
            nonmember_val_features["alora_stepped"], nonmember_val_features["alora_base"] = feature_computer.compute_features(nonmember_val_tokens_dl,learning_rate=args.alora.learning_rate,accelerator=accelerator)            
            member_train_features["alora"] = ALoRa.reduce(member_train_features["alora_stepped"], member_train_features["alora_base"])
            member_val_features["alora"] = ALoRa.reduce(member_val_features["alora_stepped"], member_val_features["alora_base"])
            nonmember_train_features["alora"] = ALoRa.reduce(nonmember_train_features["alora_stepped"], nonmember_train_features["alora_base"])
            nonmember_val_features["alora"] = ALoRa.reduce(nonmember_val_features["alora_stepped"], nonmember_val_features["alora_base"])
            feature_computer.unload_model()
        if "gradnorm" in args.features.names:
            args.gradnorm.norms = [p if p!="inf" else float('inf') for p in args.gradnorm.norms]
            feature_computer = GradNorm(model_name=args.model.name,model_revision=args.model.revision,model_cache_dir=args.model.cache_dir)
            feature_computer.load_model()
            member_train_features["gradnorm"] = feature_computer.compute_features(member_train_tokens_dl,norms=args.gradnorm.norms,accelerator=accelerator)
            member_val_features["gradnorm"] = feature_computer.compute_features(member_val_tokens_dl,norms=args.gradnorm.norms,accelerator=accelerator)
            nonmember_train_features["gradnorm"] = feature_computer.compute_features(nonmember_train_tokens_dl,norms=args.gradnorm.norms,accelerator=accelerator)
            nonmember_val_features["gradnorm"] = feature_computer.compute_features(nonmember_val_tokens_dl,norms=args.gradnorm.norms,accelerator=accelerator)
            feature_computer.unload_model()
        if "mope" in args.features.names:
            feature_computer = MoPe(model_name=args.model.name,model_revision=args.model.revision,model_cache_dir=args.model.cache_dir)
            feature_computer.generate_new_models(tokenizer=tokenizer, num_models=args.mope.num_models, noise_stdev=args.mope.noise_stdev, noise_type=args.mope.noise_type)
            member_train_features["mope_full"] = torch.zeros((args.mope.num_models+1, args.dataset.num_train_samples))  
            member_val_features["mope_full"] = torch.zeros((args.mope.num_models+1, args.dataset.num_val_samples))  
            nonmember_train_features["mope_full"] = torch.zeros((args.mope.num_models+1, args.dataset.num_train_samples))  
            nonmember_val_features["mope_full"] = torch.zeros((args.mope.num_models+1, args.dataset.num_val_samples))  
            for model_index in range(args.mope.num_models+1):
                logger.info(f"- Computing MoPe on Model {model_index+1}/{args.mope.num_models+1}")
                feature_computer.load_model(model_index)
                member_train_features["mope_full"][model_index,:] = feature_computer.compute_features(member_train_tokens_dl,accelerator=accelerator)
                member_val_features["mope_full"][model_index,:] = feature_computer.compute_features(member_val_tokens_dl,accelerator=accelerator)
                nonmember_train_features["mope_full"][model_index,:] = feature_computer.compute_features(nonmember_train_tokens_dl,accelerator=accelerator)
                nonmember_val_features["mope_full"][model_index,:] = feature_computer.compute_features(nonmember_val_tokens_dl,accelerator=accelerator)
                feature_computer.unload_model()
            member_train_features["mope"] = MoPe.reduce(member_train_features["mope_full"])
            member_val_features["mope"] = MoPe.reduce(member_val_features["mope_full"])
            nonmember_train_features["mope"] = MoPe.reduce(nonmember_train_features["mope_full"])
            nonmember_val_features["mope"] = MoPe.reduce(nonmember_val_features["mope_full"])
        if "recall" in args.features.names:
            feature_computer = ReCaLL(model_name=args.model.name,model_revision=args.model.revision,model_cache_dir=args.model.cache_dir)
            if args.recall.prefix is None:
                def get_first_n_tokens(dataset, n):
                    tokens = []
                    for example in dataset:
                        tokens.extend(example)
                        if len(tokens) >= n:
                            return torch.tensor(tokens[:n],dtype=torch.int64)
                    return torch.tensor(tokens,dtype=torch.int64)
                args.recall.prefix = get_first_n_tokens(nonmember_train_dataset["tokens"],args.recall.prefix_length)
            elif isinstance(args.recall.prefix,str):
                args.recall.prefix = tokenizer(args.recall.prefix,return_tensors="pt")["input_ids"]
            feature_computer.load_model()
            member_train_features["recall_conditional"] = feature_computer.compute_features(member_train_tokens_dl,accelerator=accelerator,prefix=args.recall.prefix)
            member_val_features["recall_conditional"] = feature_computer.compute_features(member_val_tokens_dl,accelerator=accelerator,prefix=args.recall.prefix)
            nonmember_train_features["recall_conditional"] = feature_computer.compute_features(nonmember_train_tokens_dl,accelerator=accelerator,prefix=args.recall.prefix)
            nonmember_val_features["recall_conditional"] = feature_computer.compute_features(nonmember_val_tokens_dl,accelerator=accelerator,prefix=args.recall.prefix)
            member_train_features["recall_unconditional"] = feature_computer.compute_features(member_train_tokens_dl,accelerator=accelerator)
            member_val_features["recall_unconditional"] = feature_computer.compute_features(member_val_tokens_dl,accelerator=accelerator)
            nonmember_train_features["recall_unconditional"] = feature_computer.compute_features(nonmember_train_tokens_dl,accelerator=accelerator)
            nonmember_val_features["recall_unconditional"] = feature_computer.compute_features(nonmember_val_tokens_dl,accelerator=accelerator)
            feature_computer.unload_model()
            member_train_features["recall"] = ReCaLL.reduce(member_train_features["recall_conditional"],member_train_features["recall_unconditional"])
            member_val_features["recall"] = ReCaLL.reduce(member_val_features["recall_conditional"],member_val_features["recall_unconditional"])
            nonmember_train_features["recall"] = ReCaLL.reduce(nonmember_train_features["recall_conditional"],nonmember_train_features["recall_unconditional"])
            nonmember_val_features["recall"] = ReCaLL.reduce(nonmember_val_features["recall_conditional"],nonmember_val_features["recall_unconditional"])
        if "dcpdd" in args.features.names:
            feature_computer = DCPDD(model_name=args.model.name,model_revision=args.model.revision,model_cache_dir=args.model.cache_dir)
            dcpdd_ref_dataset = load_dataset_with_metadata(args.dcpdd.dataset_name,data_dir=args.dcpdd.dataset_subset,split=args.dcpdd.dataset_split).select(range(args.dcpdd.dataset_size))
            dcpdd_ref_dl = DataLoader(dcpdd_ref_dataset["tokens"], collate_fn=lambda batch: torch.tensor(batch, dtype=torch.long))
            feature_computer.load_model()
            member_train_features["dcpdd_target"] = feature_computer.compute_features(member_train_tokens_dl,accelerator=accelerator,mode="primary")
            member_val_features["dcpdd_target"] = feature_computer.compute_features(member_val_tokens_dl,accelerator=accelerator,mode="primary")
            nonmember_train_features["dcpdd_target"] = feature_computer.compute_features(nonmember_train_tokens_dl,accelerator=accelerator,mode="primary")
            nonmember_val_features["dcpdd_target"] = feature_computer.compute_features(nonmember_val_tokens_dl,accelerator=accelerator,mode="primary")
            feature_computer.unload_model()
            member_train_features["dcpdd_ref"] = feature_computer.compute_features(dcpdd_ref_dl,tokenizer=tokenizer,mode="ref")
            member_val_features["dcpdd_ref"] = feature_computer.compute_features(dcpdd_ref_dl,tokenizer=tokenizer,mode="ref")
            nonmember_train_features["dcpdd_ref"] = feature_computer.compute_features(dcpdd_ref_dl,tokenizer=tokenizer,mode="ref")
            nonmember_val_features["dcpdd_ref"] = feature_computer.compute_features(dcpdd_ref_dl,tokenizer=tokenizer,mode="ref")
            member_train_features["dcpdd"] = DCPDD.reduce(member_train_tokens_dl,member_train_features["dcpdd_target"],member_train_features["dcpdd_ref"])
            member_val_features["dcpdd"] = DCPDD.reduce(member_val_tokens_dl,member_val_features["dcpdd_target"],member_val_features["dcpdd_ref"])
            nonmember_train_features["dcpdd"] = DCPDD.reduce(nonmember_train_tokens_dl,nonmember_train_features["dcpdd_target"],nonmember_train_features["dcpdd_ref"])
            nonmember_val_features["dcpdd"] = DCPDD.reduce(nonmember_val_tokens_dl,nonmember_val_features["dcpdd_target"],nonmember_val_features["dcpdd_ref"])
        if "modelstealing" in args.features.names:
            feature_computer = ModelStealing(model_name=args.model.name,model_revision=args.model.revision,model_cache_dir=args.model.cache_dir)
            feature_computer.load_model()
            # Load some random internet text
            svd_dataset = ThePile.load_val(number=next(feature_computer.model.parameters()).shape[1], seed=314159, tokenizer=tokenizer)["tokens"]
            svd_dataloader = DataLoader(svd_dataset, batch_size=1, collate_fn=lambda batch: torch.tensor(batch, dtype=torch.long))
            svd_embedding_projection_layer, projector = feature_computer.prepare_projection(svd_dataloader=svd_dataloader,proj_type=args.modelstealing.project_type,proj_dim=args.modelstealing.proj_dim_last,proj_seed=args.modelstealing.proj_seed,device=accelerator.device)
            member_train_features["modelstealing"] = feature_computer.compute_features(member_train_tokens_dl,svd_embedding_projection_layer=svd_embedding_projection_layer,projector=projector,device=accelerator.device)
            member_val_features["modelstealing"] = feature_computer.compute_features(member_val_tokens_dl,svd_embedding_projection_layer=svd_embedding_projection_layer,projector=projector,device=accelerator.device)
            nonmember_train_features["modelstealing"] = feature_computer.compute_features(nonmember_train_tokens_dl,svd_embedding_projection_layer=svd_embedding_projection_layer,projector=projector,device=accelerator.device)
            nonmember_val_features["modelstealing"] = feature_computer.compute_features(nonmember_val_tokens_dl,svd_embedding_projection_layer=svd_embedding_projection_layer,projector=projector,device=accelerator.device)
            feature_computer.unload_model()
        if "jl" in args.features.names:
            feature_computer = JL(model_name=args.model.name,model_revision=args.model.revision,model_cache_dir=args.model.cache_dir)
            feature_computer.load_model()
            member_train_features["jl"] = feature_computer.compute_features(member_train_tokens_dl,proj_dim_x=args.jl.proj_dim_x,proj_dim_layer=args.jl.proj_dim_layer,proj_dim_group=args.jl.proj_dim_group,proj_type=args.jl.proj_type,proj_seed=args.jl.proj_seed,device=accelerator.device,mode=args.jl.mode,num_splits=args.jl.num_splits)
            member_val_features["jl"] = feature_computer.compute_features(member_val_tokens_dl,proj_dim_x=args.jl.proj_dim_x,proj_dim_layer=args.jl.proj_dim_layer,proj_dim_group=args.jl.proj_dim_group,proj_type=args.jl.proj_type,proj_seed=args.jl.proj_seed,device=accelerator.device,mode=args.jl.mode,num_splits=args.jl.num_splits)
            nonmember_train_features["jl"] = feature_computer.compute_features(nonmember_train_tokens_dl,proj_dim_x=args.jl.proj_dim_x,proj_dim_layer=args.jl.proj_dim_layer,proj_dim_group=args.jl.proj_dim_group,proj_type=args.jl.proj_type,proj_seed=args.jl.proj_seed,device=accelerator.device,mode=args.jl.mode,num_splits=args.jl.num_splits)
            nonmember_val_features["jl"] = feature_computer.compute_features(nonmember_val_tokens_dl,proj_dim_x=args.jl.proj_dim_x,proj_dim_layer=args.jl.proj_dim_layer,proj_dim_group=args.jl.proj_dim_group,proj_type=args.jl.proj_type,proj_seed=args.jl.proj_seed,device=accelerator.device,mode=args.jl.mode,num_splits=args.jl.num_splits)
            feature_computer.unload_model()
        if "detectgpt" in args.features.names:
            feature_computer = DetectGPT(model_name=args.model.name,model_revision=args.model.revision,model_cache_dir=args.model.cache_dir)
            feature_computer.load_model()
            detect_args = {'buffer_size':1, 'mask_top_p': 10, 'pct_words_masked':.15, 'span_length':2, 'num_perts': args.detectgpt.num_perts, 'device': accelerator.device, "model_max_length": config.max_position_embeddings}
            member_train_features["detectgpt"] = feature_computer.compute_features(member_train_tokens_dl,device=accelerator.device,detect_args=detect_args)
            member_val_features["detectgpt"] = feature_computer.compute_features(member_val_tokens_dl,device=accelerator.device,detect_args=detect_args)
            nonmember_train_features["detectgpt"] = feature_computer.compute_features(nonmember_train_tokens_dl,device=accelerator.device,detect_args=detect_args)
            nonmember_val_features["detectgpt"] = feature_computer.compute_features(nonmember_val_tokens_dl,device=accelerator.device,detect_args=detect_args)
            feature_computer.unload_model()
        if "quantile" in args.features.names:
            feature_computer = Quantile(model_name=args.model.name,model_revision=args.model.revision,model_cache_dir=args.model.cache_dir,
                                         ref_model_name=args.quantile.ref_name,ref_model_revision=args.quantile.ref_revision,ref_model_cache_dir=args.quantile.ref_cache_dir)
            
            feature_computer.load_model(0)
            member_train_features["quantile_primary"] = feature_computer.compute_features(member_train_tokens_dl,accelerator=accelerator,mode="primary")
            member_val_features["quantile_primary"] = feature_computer.compute_features(member_val_tokens_dl,accelerator=accelerator,mode="primary")
            nonmember_train_features["quantile_primary"] = feature_computer.compute_features(nonmember_train_tokens_dl,accelerator=accelerator,mode="primary")
            nonmember_val_features["quantile_primary"] = feature_computer.compute_features(nonmember_val_tokens_dl,accelerator=accelerator,mode="primary")
            feature_computer.unload_model()
            member_train_features["quantile_ref"] = torch.zeros((args.dataset.num_train_samples, args.quantile.num_ref, 2))
            member_val_features["quantile_ref"] = torch.zeros((args.dataset.num_val_samples, args.quantile.num_ref, 2))
            nonmember_train_features["quantile_ref"] = torch.zeros((args.dataset.num_train_samples, args.quantile.num_ref, 2))
            nonmember_val_features["quantile_ref"] = torch.zeros((args.dataset.num_val_samples, args.quantile.num_ref, 2))
            for i in range(args.quantile.num_ref):
                feature_computer.finetune_ref(nonmember_train_tokens=torch.tensor(nonmember_train_dataset["tokens"],dtype=torch.int64),neg_log_probs=nonmember_train_features["quantile_primary"],accelerator=accelerator)
                member_train_features["quantile_ref"][:,i,:] = feature_computer.compute_features(member_train_tokens_dl,accelerator=accelerator,mode="ref")
                member_val_features["quantile_ref"][:,i,:] = feature_computer.compute_features(member_val_tokens_dl,accelerator=accelerator,mode="ref")
                nonmember_train_features["quantile_ref"][:,i,:] = feature_computer.compute_features(nonmember_train_tokens_dl,accelerator=accelerator,mode="ref")
                nonmember_val_features["quantile_ref"][:,i,:] = feature_computer.compute_features(nonmember_val_tokens_dl,accelerator=accelerator,mode="ref")
                feature_computer.unload_model()
            member_train_features["quantile"] = Quantile.reduce(member_train_features["quantile_primary"],member_train_features["quantile_ref"])
            member_val_features["quantile"] = Quantile.reduce(member_val_features["quantile_primary"],member_val_features["quantile_ref"])
            nonmember_train_features["quantile"] = Quantile.reduce(nonmember_train_features["quantile_primary"],nonmember_train_features["quantile_ref"])
            nonmember_val_features["quantile"] = Quantile.reduce(nonmember_val_features["quantile_primary"],nonmember_val_features["quantile_ref"])

        torch.save(member_train_features,f"{args.experiment_name}_member_train.pt")
        torch.save(member_val_features,f"{args.experiment_name}_member_val.pt")
        torch.save(nonmember_train_features,f"{args.experiment_name}_nonmember_train.pt")
        torch.save(nonmember_val_features,f"{args.experiment_name}_nonmember_val.pt")
        end = time.perf_counter()
        logger.info(f"- Computing features took {end-start} seconds.")
    else:
        logger.info("Loading Features")
        start = time.perf_counter()

        member_train_features = FeatureSet.load(args.features.member_train_paths)
        member_val_features = FeatureSet.load(args.features.member_val_paths)
        nonmember_train_features = FeatureSet.load(args.features.nonmember_train_paths)
        nonmember_val_features = FeatureSet.load(args.features.nonmember_val_paths)

        end = time.perf_counter()
        logger.info(f"- Loading features took {end-start} seconds.")
    ####################################################################################################
    # 2. TRAIN CLASSIFIER
    ####################################################################################################
    if args.classifier.train:    
        logger.info("Training Classifier")
        start = time.perf_counter()

        features = {}
        print(f"member_train_features.keys() = {member_train_features.keys()}")
        for feature_name in args.features.names:
            features[feature_name] = torch.cat((member_train_features[feature_name][:args.dataset.num_train_samples],nonmember_train_features[feature_name][:args.dataset.num_train_samples]),dim=0)
        labels = torch.cat((torch.ones(len(member_train_features[feature_name][:args.dataset.num_train_samples])),torch.zeros(len(nonmember_train_features[feature_name][:args.dataset.num_train_samples]))),dim=0)
        
        if args.classifier.name=="decision_tree":
            classifier = DecisionTree(clf_name=args.classifier.path, feature_names=args.features.names)
            features, labels = classifier.preprocess_features(features,labels,fit_scaler=True)
            predictions = classifier.train_clf(features, labels, max_depth=args.decision_tree.max_depth)
        elif args.classifier.name=="random_forest":
            classifier = RandomForest(clf_name=args.classifier.path, feature_names=args.features.names)
            features, labels = classifier.preprocess_features(features,labels,fit_scaler=True)
            predictions = classifier.train_clf(features, labels, n_estimators=args.random_forest.n_estimators, max_depth=args.random_forest.max_depth)
        elif args.classifier.name=="gradient_boosting":
            classifier = GradientBoosting(clf_name=args.classifier.path, feature_names=args.features.names)
            features, labels = classifier.preprocess_features(features,labels,fit_scaler=True)
            predictions = classifier.train_clf(features, labels, max_iter=args.gradient_boosting.max_iter, max_depth=args.gradient_boosting.max_depth)
        elif args.classifier.name=="log_reg":
            classifier = LogReg(clf_name=args.classifier.path, feature_names=args.features.names)
            features, labels = classifier.preprocess_features(features,labels,fit_scaler=True)
            predictions = classifier.train_clf(features, labels, max_iter=args.log_reg.max_iter)
        elif args.classifier.name=="neural_net":
            classifier = NeuralNet(clf_name=args.classifier.path, feature_names=args.features.names)
            features, labels = classifier.preprocess_features(features,labels,fit_scaler=True)
            predictions = classifier.train_clf(features, labels, clf_size=args.neural_net.size, epochs=args.neural_net.num_epochs, batch_size=args.neural_net.batch_size, accelerator=accelerator)
        else:
            raise ValueError(f"Unrecognized classifier: {args.classifier.name}")

        os.makedirs(os.path.dirname(args.classifier.path), exist_ok=True)
        torch.save(classifier,args.classifier.path)

        plot_ROC(predictions[labels==1], predictions[labels==0], plot_title=f"{args.experiment_name}_train", log_scale=False, show_plot=False, save_name=f"{args.experiment_name}_train")
        plot_ROC(predictions[labels==1], predictions[labels==0], plot_title=f"{args.experiment_name}_train", log_scale=True, show_plot=False, save_name=f"{args.experiment_name}_train_log")
        plot_histogram(predictions[labels==1], predictions[labels==0], plot_title=f"{args.experiment_name}_train", normalize=False, show_plot=False, save_name=f"{args.experiment_name}_train")
        plot_histogram(predictions[labels==1], predictions[labels==0], plot_title=f"{args.experiment_name}_train", normalize=True, show_plot=False, save_name=f"{args.experiment_name}_train_z")

        end = time.perf_counter()
        logger.info(f"- Classifier training took {end-start} seconds.")
    else:
        logger.info("Loading Classifier")
        start = time.perf_counter()
        
        if args.classifier.name=="threshold":
            classifier = Threshold()
        else:
            classifier = torch.load(args.classifier.path)
            if classifier.feature_set!=args.features.names:
                raise ValueError("Specified feature set does not match saved feature set!")

        end = time.perf_counter()
        logger.info(f"- Loading classifier took {end-start} seconds.")
    ####################################################################################################
    # 3. GET CLASSIFICATIONS
    ####################################################################################################
    if args.classifier.infer:
        logger.info("Getting Classifications")
        start = time.perf_counter()

        member_val_features = {feature:member_val_features[feature][:args.dataset.num_val_samples] for feature in args.features.names}
        nonmember_val_features = {feature:nonmember_val_features[feature][:args.dataset.num_val_samples] for feature in args.features.names}

        member_val_features = classifier.preprocess_features(member_val_features,fit_scaler=False)
        nonmember_val_features = classifier.preprocess_features(nonmember_val_features,fit_scaler=False)

        if args.classifier.name=="neural_net":
            member_probabilities = classifier.predict_membership(member_val_features,accelerator=accelerator)
            nonmember_probabilities = classifier.predict_membership(nonmember_val_features,accelerator=accelerator)
        else:
            member_probabilities = classifier.predict_membership(member_val_features)
            nonmember_probabilities = classifier.predict_membership(nonmember_val_features)
        torch.save(member_probabilities,f"{args.experiment_name}_member.pt")
        torch.save(nonmember_probabilities,f"{args.experiment_name}_nonmember.pt")

        plot_ROC(member_probabilities, nonmember_probabilities, plot_title=args.experiment_name, log_scale=False, show_plot=False, save_name=f"{args.experiment_name}")
        plot_ROC(member_probabilities, nonmember_probabilities, plot_title=args.experiment_name, log_scale=True, show_plot=False, save_name=f"{args.experiment_name}_log")
        plot_histogram(member_probabilities, nonmember_probabilities, plot_title=args.experiment_name, normalize=False, show_plot=False, save_name=f"{args.experiment_name}")
        plot_histogram(member_probabilities, nonmember_probabilities, plot_title=args.experiment_name, normalize=True, show_plot=False, save_name=f"{args.experiment_name}_z")

        end = time.perf_counter()
        logger.info(f"- Classification took {end-start} seconds.")

    end_total = time.perf_counter()
    logger.info(f"- Experiment {args.experiment_name} took {end_total-start_total} seconds.")

if __name__ == "__main__":
    main()