#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import argparse
import warnings
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import ks_2samp

from utils_aletheia import *

warnings.filterwarnings('ignore')


def main(args):
    setup_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading and preprocessing dataset: {args.dataset.upper()}")
    if args.dataset == 'nsl':
        # NSL-KDD的列名是固定的，共42列（41个特征+1个标签）
        col_names = [str(i) for i in range(41)] + ['label']
        Train_data = pd.read_csv(args.nsl_train_path, header=None, names=col_names)
        Test_data = pd.read_csv(args.nsl_test_path, header=None, names=col_names)
        splitter = SplitData(dataset_name='nsl')
        x_train_raw, y_train_raw, feature_names = splitter.transform(Train_data)
        x_test_raw, y_test_raw, _ = splitter.transform(Test_data)
        input_dim = x_train_raw.shape[1]
    elif args.dataset == 'unsw':
        Train_data = load_data(args.unsw_train_path)
        Test_data = load_data(args.unsw_test_path)
        # 合并训练和测试集以确保One-Hot编码一致性
        full_data = pd.concat([Train_data, Test_data], ignore_index=True)
        splitter = SplitData(dataset_name='unsw')
        x_full, y_full, feature_names = splitter.transform(full_data)
        # 拆分回训练和测试
        train_size = len(Train_data)
        x_train_raw, y_train_raw = x_full[:train_size], y_full[:train_size]
        x_test_raw, y_test_raw = x_full[train_size:], y_full[train_size:]
        input_dim = x_train_raw.shape[1]
    else:
        raise ValueError("Dataset not supported. Choose 'nsl' or 'unsw'.")

    print(f"Dataset loaded. Input dimension after preprocessing: {input_dim}")


    feature_groups, feature_to_group = feature_grouper(feature_names, args.dataset)
    print("Feature groups created.")

    x_train, y_train = torch.FloatTensor(x_train_raw), torch.FloatTensor(y_train_raw)
    x_test, y_test = torch.FloatTensor(x_test_raw), torch.FloatTensor(y_test_raw)

    online_x_train, online_x_stream, online_y_train, online_y_stream = train_test_split(
        x_train, y_train, test_size=(1 - args.percent), random_state=args.seed
    )

    online_x_stream = torch.cat((online_x_stream, x_test), dim=0)
    online_y_stream = torch.cat((online_y_stream, y_test), dim=0)
    print(f"Initial training data size: {online_x_train.shape[0]}")
    print(f"Online stream data size: {online_x_stream.shape[0]}")

    model = AE_classifier(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_ds = TensorDataset(online_x_train, online_y_train)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True)

    model.train()
    for epoch in range(args.epochs):
        for data, labels in train_loader:
            inputs, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            _, _, classifications = model(inputs)
            loss = criterion(classifications.squeeze(), labels)
            loss.backward()
            optimizer.step()

    vae_model = VAE(input_dim=input_dim).to(device)
    vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.001)
    vae_model.train()
    for epoch in range(args.epochs):
        for data, _ in train_loader:
            inputs = data.to(device)
            vae_optimizer.zero_grad()
            recon_batch, mu, logvar = vae_model(inputs)
            loss = vae_loss_function(recon_batch, inputs, mu, logvar)
            loss.backward()
            vae_optimizer.step()

    print("\n" + "=" * 20 + " Phase 2: Starting Online Continual Learning " + "=" * 20)
    memory_buffer_x = online_x_train.clone().to(device)
    memory_buffer_y = online_y_train.clone().to(device)

    for i in range(0, online_x_stream.shape[0], args.sample_interval):
        batch_num = i // args.sample_interval + 1
        print(f"\n===== Processing Online Batch {batch_num} =====")

        new_data_x = online_x_stream[i:i + args.sample_interval].to(device)
        new_data_y = online_y_stream[i:i + args.sample_interval].to(device)
        if len(new_data_x) == 0: continue

        print("--- Running Drift Monitor ---")
        model.eval()
        vae_model.eval()

        with torch.no_grad():
            _, _, old_logits = model(memory_buffer_x)
            _, _, new_logits = model(new_data_x)
            _, p_value_model = ks_2samp(old_logits.squeeze().cpu().numpy(), new_logits.squeeze().cpu().numpy())

            recon_old, _, _ = vae_model(memory_buffer_x)
            error_old = torch.mean((memory_buffer_x - recon_old) ** 2, axis=1)
            recon_new, _, _ = vae_model(new_data_x)
            error_new = torch.mean((new_data_x - recon_new) ** 2, axis=1)
            _, p_value_data = ks_2samp(error_old.cpu().numpy(), error_new.cpu().numpy())

            uncertainty_old = get_mc_dropout_uncertainty(model, memory_buffer_x)
            uncertainty_new = get_mc_dropout_uncertainty(model, new_data_x)
            _, p_value_uncertainty = ks_2samp(uncertainty_old.cpu().numpy(), uncertainty_new.cpu().numpy())

        drift_signals = [p < args.drift_threshold for p in [p_value_model, p_value_data, p_value_uncertainty]]
        drift_consensus = sum(drift_signals) >= 2

        if drift_consensus:
            # some core code wiil be open-sources after accepted! thanks!

            print("--- Running Drift Analyst ---")
            attribution_scores = run_fused_attribution_hybrid(model, vae_model, new_data_x, memory_buffer_x, device)
            target_lora_layers = map_drift_to_layers(attribution_scores, feature_groups)

            print("--- Updating Memory Buffer with Active Learning & Archiving ---")
            old_uncertainties = get_mc_dropout_uncertainty(model, memory_buffer_x)
            new_uncertainties = get_mc_dropout_uncertainty(model, new_data_x)
            add_indices = torch.topk(new_uncertainties, k=args.k_label, largest=True).indices

            keep_mask = torch.ones(len(memory_buffer_x), dtype=bool, device=device)
            memory_buffer_x = torch.cat([memory_buffer_x[keep_mask], new_data_x[add_indices]])
            memory_buffer_y = torch.cat([memory_buffer_y[keep_mask], new_data_y[add_indices]])
            print(f"Added {len(add_indices)} new samples via active learning. New buffer size: {len(memory_buffer_x)}")
            print(f"--- Applying Targeted LoRA to layers: {target_lora_layers} ---")
            lora_model = apply_lora_to_model(model, target_lora_layers)

            lora_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=0.005)
            online_train_ds = TensorDataset(memory_buffer_x, memory_buffer_y)
            online_train_loader = DataLoader(online_train_ds, batch_size=args.bs, shuffle=True)

    model.to(device).eval()
    final_test_ds = TensorDataset(x_test.to(device), y_test.to(device))
    final_test_loader = DataLoader(final_test_ds, batch_size=args.bs, shuffle=False)
    evaluate_classifier(model, final_test_loader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aletheia Framework for IDS')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--nsl_train_path", type=str)
    parser.add_argument("--nsl_test_path", type=str)
    parser.add_argument("--unsw_train_path", type=str)
    parser.add_argument("--unsw_test_path", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--epoch_1", type=int)
    parser.add_argument("--percent", type=float)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--sample_interval", type=int)
    parser.add_argument("--k_label", type=int)
    parser.add_argument("--drift_threshold", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cuda", type=str)

    args = parser.parse_args()
    main(args)