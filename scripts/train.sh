#!/bin/bash

# Baseline 3
HYDRA_FULL_ERROR=1 python src/train_v2.py -m version=6 +cluster=reg_mtg_CAL500,reg_mtg_emotify,reg_CAL500_emotify model.regularization_alpha=2.5 trainer.max_epochs=75 data.train_combine_mode=max_size_cycle +model.regularization_v2=false || true;
# HYDRA_FULL_ERROR=1 python src/train_v2.py -m version=6 +cluster=reg_mtg_CAL500,reg_mtg_emotify,reg_CAL500_emotify model.regularization_alpha=2.5,1 trainer.max_epochs=150 data.train_combine_mode=max_size_cycle model.net.num_layers=2,3 || true;

# # Baseline 3
# HYDRA_FULL_ERROR=1 python src/train_v2.py -m version=6 +cluster=reg_mtg_CAL500,reg_mtg_emotify,reg_CAL500_emotify model.regularization_alpha=0.5 trainer.max_epochs=75 data.train_combine_mode=max_size_cycle model.net.num_layers=2,3 || true;

# Baseline 1
HYDRA_FULL_ERROR=1 python src/train_v2.py -m version=6 +experiment=multirun_many_aug trainer.max_epochs=75 || true;

# Baseline 2
HYDRA_FULL_ERROR=1 python src/train_v2.py -m version=6 +cluster=mtg_CAL500,mtg_emotify,CAL500_emotify trainer.max_epochs=75 || true;

# HYDRA_FULL_ERROR=1 python src/train_v2.py -m version=5 +experiment=multirun_many || true;
# HYDRA_FULL_ERROR=1 python src/train_v2.py -m version=5 +cluster="glob(*,exclude=default)" model.net.num_layers=3 || true;
# HYDRA_FULL_ERROR=1 python src/train_v2.py -m version=5 +experiment=multirun_many_aug model.net.num_layers=3 || true;
# HYDRA_FULL_ERROR=1 python src/train_v2.py -m version=5 +experiment=multirun_many model.net.num_layers=3 || true;

# HYDRA_FULL_ERROR=1 python src/train_v2.py -m version=4 +cluster="glob(*,exclude=default)" loss_type=mse || true;
# HYDRA_FULL_ERROR=1 python src/train_v2.py -m version=4 +experiment=multirun_many_aug loss_type=mse || true;
# HYDRA_FULL_ERROR=1 python src/train_v2.py -m version=4 +experiment=multirun_many loss_type=mse || true;
