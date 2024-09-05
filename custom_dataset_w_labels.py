import os
import numpy as np
import argparse
import seaborn as sn
import pandas as pd
from datetime import datetime
import sklearn
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from bams.data import KeypointsDataset
from bams.models import BAMS
from bams import HoALoss, compute_representations, train_linear_classfier, train_linear_regressor


def load_data(path, output_folder, create_csv = True):
    '''
    Load and format keypoint data. Output should be in the shape (n_samples, seq_len, num_feats). 
    Collapse xy coordinates into the single num_feats dimension.
    '''

    import os
    import numpy as np
    import pandas as pd

    all_keypoints_reshaped = []
    video_names = []
    labels = []
    chunks_per_video = []
    minutes = 2
    fps = 30

    frames = minutes * fps * 60

    # Iterate through each subfolder in the main folder
    for subfolder in os.listdir(path):
        subfolder_path = os.path.join(path, subfolder)
        if os.path.isdir(subfolder_path):
            # Iterate through each file in the subfolder
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(subfolder_path, filename)
                    
                    # Extract video name (assuming the filename is the video name)
                    video_name = os.path.splitext(filename)[0]
                    print(f"Processing video: {video_name}")

                    # Load data from CSV. Skip the first three rows to just get the x,y data
                    df = pd.read_csv(file_path, skiprows=3, header=None, low_memory=False)
                    #midpoint = len(df) // 4

                    # Skip the first column (frame number)
                    keypoint_data = df.iloc[:, 1:]

                    # Get x and y
                    x = keypoint_data.values[:, 0::3]
                    y = keypoint_data.values[:, 1::3]
                    likelihoods = keypoint_data.values[:, 2::3]

                    mask = likelihoods >= 0.1

                    x_mean = np.array([np.nanmean(np.where(mask_row, x_row, x_row)) if not np.any(mask_row) else np.nanmean(np.where(mask_row, x_row, np.nan))
                                    for mask_row, x_row in zip(mask, x)])
                    y_mean = np.array([np.nanmean(np.where(mask_row, y_row, y_row)) if not np.any(mask_row) else np.nanmean(np.where(mask_row, y_row, np.nan))
                                    for mask_row, y_row in zip(mask, y)])

                    x = x - x_mean[:, np.newaxis]
                    y = y - y_mean[:, np.newaxis]

                    print("x: ", x)
                    print("y: ", y)

                    # Stack x and y coordinates
                    keypoints = np.stack((x, y), axis=-1)

                    # Calculate the number of samples and features
                    n_samples = keypoints.shape[0]
                    num_keypoints = keypoints.shape[1]
                    num_feats = keypoints.shape[2]

                    # Debugging 
                    
                    print(f"Number of samples: {n_samples}")
                    print(f"Number of keypoints: {num_keypoints}")
                    print(f"keypoints shape: {np.shape(keypoints)}")
                    print(f"Number of features: {num_feats}")

                    chunk_count = 0

                    # Break keypoints into chunks of size `frames`
                    for start in range(0, n_samples, frames):
                        end = start + frames
                        if end > n_samples:
                            break
                        chunk = keypoints[start:end]
                        keypoints_reshaped = chunk.reshape((frames, num_keypoints * num_feats))
                        all_keypoints_reshaped.append(keypoints_reshaped)
                        video_names.append(video_name)
                        labels.append(subfolder)  # Append the name of the subfolder to labels
                        chunk_count += 1
                    
                    #print(f"keypoints_reshaped shape: {np.shape(keypoints_reshaped)}")
                    
                    print(f"Number of chunks: {chunk_count}")
                    chunks_per_video.extend([chunk_count] * chunk_count)

    # Convert list to numpy array
    all_keypoints_reshaped = np.array(all_keypoints_reshaped)
    
    print(f"All keypoints reshaped shape: {all_keypoints_reshaped.shape}")
    #print("All keypoints reshaped:")
    #print(all_keypoints_reshaped)

    if create_csv:
        print("CREATING CSV")
    # Write video_names and labels to a CSV file
        df_labels = pd.DataFrame({
            'video_name': video_names,
            'label': labels,
            'chunks_per_video': chunks_per_video
        })
        df_labels.to_csv(os.path.join(output_folder, 'video_labels.csv'), index=False)
        print("labels written to: ", os.path.join(output_folder, 'video_labels.csv'))
    
    #input("Press Enter to proceed to training or CTRL+C to cancel")

    return all_keypoints_reshaped
    
    # keypoints = ...
    # return keypoints


def load_annotations(path):
    '''
    load labels/annotations in the following dictionary format:
    annotations = {'video_name': [str], 'label1': [int/float], 'label2': [int/float], ...}
    
    Your labels can have any name. The video_name key is optional, and is used to keep track of the video name for each sample.

    In addition, create an eval_utils dictionary with the following format:
    eval_utils = {'classification_tags': [str], 'regression_tags': [str], 'sequence_level_dict': {'label1': True/False, 'label2': True/False, ...}}

    This dictionary contains the necessary metadata for evaluating the model. The classification_tags list contains the names of all
    classification labels, the regression_tags list contains the names of all regression labels, and the sequence_level_dict contains
    the names of all labels and whether they are sequence level or not. Enter True if the label is a sequence level label, and False 
    if it is frame level. Ensure the label names in the classification_tags and regression_tags lists match the names of the labels in
    the annotations dictionary.
    '''
    df = pd.read_csv(path)
    
    num_segments = len(df)
    unique_labels = df['label'].unique()
    label_mapping = {label: index for index, label in enumerate(unique_labels)}

    annotations = {
        'video_name': df['label'].tolist(),
        'label1': [label_mapping[label] for label in df['label']]
    }

    eval_utils = {
        'classification_tags': ['label1'],
        'regression_tags': ['label1'],
        'sequence_level_dict': {'label1': True}
    }
    return annotations, eval_utils

def train(model, device, loader, optimizer, criterion, writer, step, log_every_step):
    model.train()

    for data in tqdm(loader, position=1, leave=False):
        # todo convert to float
        input = data["input"].float().to(device)  # (B, N, L)
        target = data["target_hist"].float().to(device)
        ignore_weights = data["ignore_weights"].to(device)

        # forward pass
        optimizer.zero_grad()
        embs, hoa_pred, byol_preds = model(input)

        # prediction task
        hoa_loss = criterion(target, hoa_pred, ignore_weights)

        # contrastive loss: short term
        batch_size, sequence_length, emb_dim = embs["short_term"].size()
        skip_frames, delta = 60, 5
        view_1_id = (
            torch.randint(sequence_length - skip_frames - delta, (batch_size,))
            + skip_frames
        )
        view_2_id = torch.randint(delta + 1, (batch_size,)) + view_1_id
        view_2_id = torch.clip(view_2_id, 0, sequence_length)

        view_1 = byol_preds["short_term"][torch.arange(batch_size), view_1_id]
        view_2 = embs["short_term"][torch.arange(batch_size), view_2_id]

        byol_loss_short_term = (
            1 - F.cosine_similarity(view_1, view_2.clone().detach(), dim=-1).mean()
        )

        # contrastive loss: long term
        batch_size, sequence_length, emb_dim = embs["long_term"].size()
        skip_frames = 100
        view_1_id = (
            torch.randint(sequence_length - skip_frames, (batch_size,)) + skip_frames
        )
        view_2_id = (
            torch.randint(sequence_length - skip_frames, (batch_size,)) + skip_frames
        )

        view_1 = byol_preds["long_term"][torch.arange(batch_size), view_1_id]
        view_2 = embs["long_term"][torch.arange(batch_size), view_2_id]

        byol_loss_long_term = (
            1 - F.cosine_similarity(view_1, view_2.clone().detach(), dim=-1).mean()
        )

        # backprop
        loss = 5e2 * hoa_loss + 0.5 * byol_loss_short_term + 0.5 * byol_loss_long_term

        loss.backward()
        optimizer.step()

        step += 1
        if step % log_every_step == 0:
            writer.add_scalar("train/hoa_loss", hoa_loss.item(), step)
            writer.add_scalar(
                "train/byol_loss_short_term", byol_loss_short_term.item(), step
            )
            writer.add_scalar(
                "train/byol_loss_long_term", byol_loss_long_term.item(), step
            )
            writer.add_scalar("train/total_loss", loss.item(), step)

    return step

def test(model, device, dataset, writer, epoch):
    test_idx, train_idx = train_test_split(np.arange(len(dataset)), test_size=0.8, random_state=42)
    # get embeddings
    embeddings = compute_representations(model, dataset, device)
    emb_keys = ['short_term', 'long_term']
    # decode from all embeddings
    def decode_class(keys, target, global_pool=False):
        if len(keys) == 1:
            emb = embeddings[keys[0]]
        else:
            emb = torch.cat([embeddings[key] for key in keys], dim=2)
        emb_size = emb.size(2)

        if global_pool:
            emb = torch.mean(emb, dim=1, keepdim=True)

        train_data = [emb[train_idx].reshape(-1, emb_size), target[train_idx].reshape(-1)]
        test_data = [emb[test_idx].reshape(-1, emb_size), target[test_idx].reshape(-1)]
        f1_score, cm = train_linear_classfier(target.max()+1, train_data, test_data, device, lr=1e-2, weight_decay=1e-4)
        return f1_score, cm

    def decode_scalar(keys, target, global_pool=False):
        if len(keys) == 1:
            emb = embeddings[keys[0]]
        else:
            emb = torch.cat([embeddings[key] for key in keys], dim=2)
        emb_size = emb.size(2)

        if global_pool:
            emb = torch.mean(emb, dim=1, keepdim=True)

        train_data = [emb[train_idx].reshape(-1, emb_size), target[train_idx].reshape(-1, 1)]
        test_data = [emb[test_idx].reshape(-1, emb_size), target[test_idx].reshape(-1, 1)]
        mse = train_linear_regressor(train_data, test_data, device, lr=1e-2, weight_decay=1e-4)
        return mse
    
    keys = list(dataset.annotations.keys())
    data = {}
    for batch in DataLoader(dataset, batch_size=4, shuffle=False):
      for key in keys:
        if len(data.keys()) < len(keys):
          data[key] = batch[key]
        else:
          data[key] = torch.cat([data[key], batch[key]], axis=0)

    for target_tag in dataset.eval_utils['classification_tags']:
        target = data[target_tag].type(torch.LongTensor)
        global_pool = dataset.eval_utils['sequence_level_dict'][target_tag]
        f1_score, cm = decode_class(emb_keys, target, global_pool=global_pool)
        emb_tag = '_'.join(emb_keys)
        writer.add_scalar(f'test/f1_{target_tag}_{emb_tag}', f1_score, epoch)
        writer.add_figure(f'{target_tag}_{emb_tag}', sn.heatmap(pd.DataFrame(cm), annot=True).get_figure(), epoch)

    for target_tag in dataset.eval_utils['regression_tags']:
        target = torch.FloatTensor(data[target_tag].float())
        global_pool = dataset.eval_utils['sequence_level_dict'][target_tag]
        mse = decode_scalar(emb_keys, target, global_pool=global_pool)
        emb_tag = '_'.join(emb_keys)
        writer.add_scalar(f'test/mse_{target_tag}_{emb_tag}', mse, epoch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/mabe")
    parser.add_argument("--cache_path", type=str, default="./data/mabe/custom_dataset")
    parser.add_argument("--hoa_bins", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--log_every_step", type=int, default=50)
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing the keypoint data.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    keypoints = load_data(args.csv_path)
    num_sequences = keypoints.shape[0]
    annotations, eval_utils = load_annotations(args.data_root, num_sequences)

    dataset = KeypointsDataset(
        keypoints=keypoints,
        hoa_bins=args.hoa_bins,
        cache_path=args.cache_path,
        cache=False,
        annotations=annotations,
        eval_utils=eval_utils
    )

    print("Number of sequences:", len(dataset))
    print("Keypoints shape:", keypoints.shape)
    print("Annotations:", annotations)

    # prepare dataloaders
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # build model
    model = BAMS(
        input_size=dataset.input_size,
        short_term=dict(num_channels=(64, 64, 64, 64), kernel_size=3),
        long_term=dict(num_channels=(64, 64, 64, 64, 64), kernel_size=3, dilation=4),
        predictor=dict(
            hidden_layers=(-1, 256, 512, 512, dataset.target_size * args.hoa_bins)
        ),
    ).to(device)

    print(model)

    model_name = f"bams-custom-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    writer = SummaryWriter("runs/" + model_name)

    main_params = [p for name, p in model.named_parameters() if "byol" not in name]
    byol_params = list(model.byol_predictors.parameters())

    optimizer = optim.AdamW(
        [{"params": main_params}, {"params": byol_params, "lr": args.lr * 10}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)
    criterion = HoALoss(hoa_bins=args.hoa_bins, skip_frames=100)

    step = 0
    for epoch in tqdm(range(1, args.epochs + 1), position=0):
        step = train(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            writer,
            step,
            args.log_every_step,
        )
        scheduler.step()
        if epoch % 50 == 1:
            torch.save(model.state_dict(), model_name + ".pt")
            test(model, device, dataset, writer, epoch)


if __name__ == "__main__":
    main()