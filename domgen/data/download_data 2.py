import argparse
import tarfile
from zipfile import ZipFile
import gdown
from wilds import get_dataset
import os
import pandas as pd
import shutil


def download_camelyon(datadir):
    if not os.path.exists(os.path.join(datadir, 'camelyon17')):
        confirmation = input("The dataset you're about to download is very large (15GB on disk)!"
                             " Do you want to continue (y/n) ")
        if confirmation == 'y':
            dataset = get_dataset(dataset="camelyon17", download=True, root_dir=f"{datadir}/camelyon17")
            df = pd.read_csv(f'{datadir}/camelyon17/camelyon17_v1.0/metadata.csv')

            center0 = df[df['center'] == 0]
            center1 = df[df['center'] == 1]
            center2 = df[df['center'] == 2]
            center3 = df[df['center'] == 3]
            center4 = df[df['center'] == 4]

            center0_tumor_0 = center0[center0['tumor'] == 0]
            center0_tumor_1 = center0[center0['tumor'] == 1]

            center1_tumor_0 = center1[center1['tumor'] == 0]
            center1_tumor_1 = center1[center1['tumor'] == 1]

            center2_tumor_0 = center2[center2['tumor'] == 0]
            center2_tumor_1 = center2[center2['tumor'] == 1]

            center3_tumor_0 = center3[center3['tumor'] == 0]
            center3_tumor_1 = center3[center3['tumor'] == 1]

            center4_tumor_0 = center4[center4['tumor'] == 0]
            center4_tumor_1 = center4[center4['tumor'] == 1]

            base_dir = f"{datadir}/camelyon17"

            center_dfs = {
                0: (center0, center0_tumor_0, center0_tumor_1),
                1: (center1, center1_tumor_0, center1_tumor_1),
                2: (center2, center2_tumor_0, center2_tumor_1),
                3: (center3, center3_tumor_0, center3_tumor_1),
                4: (center4, center4_tumor_0, center4_tumor_1),
            }

            for center, (df, df_tumor_0, df_tumor_1) in center_dfs.items():
                center_dir = os.path.join(base_dir, f"center_{center}")
                tumor_0_dir = os.path.join(center_dir, "tumor_0")
                tumor_1_dir = os.path.join(center_dir, "tumor_1")

                os.makedirs(tumor_0_dir, exist_ok=True)
                os.makedirs(tumor_1_dir, exist_ok=True)

                df.to_csv(os.path.join(center_dir, f"center_{center}.csv"), index=False)
                df_tumor_0.to_csv(os.path.join(tumor_0_dir, "tumor_0.csv"), index=False)
                df_tumor_1.to_csv(os.path.join(tumor_1_dir, "tumor_1.csv"), index=False)

                print(f"Data for center {center} saved in {center_dir}")

            print("BUILDING DOMAINS...")
            patches_dir = f"{datadir}/camelyon17/camelyon17_v1.0/patches"

            for center in range(0, 5):
                center_dir = os.path.join(base_dir, f"center_{center}")

                for tumor_type in ['tumor_0', 'tumor_1']:
                    csv_path = os.path.join(center_dir, tumor_type, f"{tumor_type}.csv")
                    if not os.path.exists(csv_path):
                        print(f"CSV file not found: {csv_path}")
                        continue

                    df = pd.read_csv(csv_path)

                    for _, row in df.iterrows():
                        patient = f"{int(row['patient']):03}"
                        node = row['node']
                        x_coord = row['x_coord']
                        y_coord = row['y_coord']

                        source_dir = os.path.join(patches_dir, f"patient_{patient}_node_{node}")
                        if not os.path.exists(source_dir):
                            print(f"Source directory not found: {source_dir}")
                            continue

                        image_name = f"patch_patient_{patient}_node_{node}_x_{x_coord}_y_{y_coord}.png"
                        source_path = os.path.join(source_dir, image_name)
                        if not os.path.exists(source_path):
                            print(f"Image file not found: {source_path}")
                            continue

                        target_dir = os.path.join(center_dir, tumor_type)
                        target_path = os.path.join(target_dir, image_name)

                        shutil.move(source_path, target_path)
            print(f"Moved images")

            print('CLEANING UP...')
            directory_to_delete = f"{datadir}/camelyon17/camelyon17_v1.0"

            if os.path.exists(directory_to_delete) and os.path.isdir(directory_to_delete):
                shutil.rmtree(directory_to_delete)
                print(f"Directory '{directory_to_delete}' and all its contents have been deleted.")
            else:
                print(f"Directory '{directory_to_delete}' does not exist.")
        else:
            print(f"Understandable, goodbye!")


# the downloading pacs + stage_path + download and extract functions are taken directly from:
# https://github.com/facebookresearch/DomainBed/blob/main/domainbed/scripts/download.py
def download_pacs(data_dir):
    # Original URL: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    full_path = stage_path(data_dir, "PACS")

    download_and_extract("https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd",
                         os.path.join(data_dir, "PACS.zip"))

    os.rename(os.path.join(data_dir, "kfold"),
              full_path)


def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='./datasets')
    parser.add_argument('--download_camelyon', action='store_true', default=False)
    parser.add_argument('--download_pacs', action='store_true', default=False)
    parser.add_argument('--all', action='store_true', default=False)
    args = parser.parse_args()

    if args.all:
        args.download_camelyon = True
        args.download_pacs = True
    if args.download_camelyon:
        download_camelyon(args.datadir)
    if args.download_pacs:
        download_pacs(args.datadir)
