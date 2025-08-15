import tensorflow as tf
import pandas as pd
import numpy as np

def make_tf_dataset(dataframe, batch_size=None):
    attack_columns = [
        'Type of attack_ARP Spoofing',
        'Type of attack_DoS Attack',
        'Type of attack_Nmap Port Scan',
        'Type of attack_No Attack',
        'Type of attack_Smurf Attack'
    ]

    print("\n📋 Class distribution BEFORE make_tf_dataset:")
    for col in attack_columns:
        count = dataframe[col].sum()
        print(f"🔹 {col.replace('Type of attack_', '')}: {int(count)} samples")

    # Get the class with the maximum number of samples
    max_samples = int(dataframe[attack_columns].sum().max())

    balanced_dataframes = []

    for attack in attack_columns:
        attack_df = dataframe[dataframe[attack] == 1]

        # Oversample (with replacement if needed)
        attack_df = attack_df.sample(n=max_samples, replace=True, random_state=42)

        balanced_dataframes.append(attack_df)

    balanced_df = pd.concat(balanced_dataframes, ignore_index=True)

    print("\n📋 Class distribution AFTER balancing:")
    for i, col in enumerate(attack_columns):
        print(f"🔹 {col.replace('Type of attack_', '')}: {max_samples} samples")

    y = balanced_df[attack_columns].values

    # حذف الأعمدة من الدخل
    balanced_df = balanced_df.drop(columns=['Type'] + attack_columns)

    y = tf.cast(y, tf.int64)

    dataset = tf.data.Dataset.from_tensor_slices((balanced_df.values, y))
    dataset = dataset.shuffle(buffer_size=len(balanced_df), seed=2048)

    print(f"\n🔄 Total shuffled samples in dataset: {len(balanced_df)}")

    if batch_size:
        dataset = dataset.batch(batch_size)

    return dataset
