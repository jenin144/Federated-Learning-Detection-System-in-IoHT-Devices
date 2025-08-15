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
    present_classes = {}
    for col in attack_columns:
        count = dataframe[col].sum()
        print(f"🔹 {col.replace('Type of attack_', '')}: {int(count)} samples")
        if count > 0:
            present_classes[col] = count

    # Exit early if no data
    if not present_classes:
        print("⚠️ No valid classes found. Returning empty dataset.")
        return tf.data.Dataset.from_tensor_slices((np.array([]), np.array([])))

    max_samples = max(present_classes.values())
    balanced_dataframes = []

    for attack, count in present_classes.items():
        attack_df = dataframe[dataframe[attack] == 1]
        
        # Only oversample if we have samples
        if count > 0:
            attack_df = attack_df.sample(
                n=max_samples, 
                replace=True, 
                random_state=42
            )
            balanced_dataframes.append(attack_df)

    balanced_df = pd.concat(balanced_dataframes, ignore_index=True)

    print("\n📋 Class distribution AFTER balancing:")
    for attack in present_classes:
        count = balanced_df[attack].sum()
        print(f"🔹 {attack.replace('Type of attack_', '')}: {int(count)} samples")

    y = balanced_df[attack_columns].values
    balanced_df = balanced_df.drop(columns=['Type'] + attack_columns)

    dataset = tf.data.Dataset.from_tensor_slices((balanced_df.values, y))
    dataset = dataset.shuffle(buffer_size=len(balanced_df), seed=2048)

    if batch_size:
        dataset = dataset.batch(batch_size)

    return dataset