import os.path

import tensorflow as tf
from utils.image import load_image, encode_image_to_bytes
from tqdm import tqdm


class TFRecordBuilder:
    def __init__(self, dataset_df, tfrecord_path=None):
        self.dataset_df = dataset_df
        self.tfrecord_path = tfrecord_path

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def serialize_example(self, image_path, mask_path):
        image = load_image(image_path, mode="L")
        mask = load_image(mask_path, mode="L")

        feature = {
            'image': self._bytes_feature(encode_image_to_bytes(image)),
            'mask': self._bytes_feature(encode_image_to_bytes(mask)),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


    def write_tfrecord(self, log_title="TFRecord"):
        if log_title == "Train":
            self.tfrecord_path = os.path.join(self.tfrecord_path, "train.tfrecord")
        elif log_title == "Validation":
            self.tfrecord_path = os.path.join(self.tfrecord_path, "val.tfrecord")
        else:
            self.tfrecord_path = os.path.join(self.tfrecord_path, "test.tfrecord")

        with tf.io.TFRecordWriter(self.tfrecord_path) as writer:
            for idx, row in tqdm(self.dataset_df.iterrows(), total=len(self.dataset_df), desc=f"Writing {log_title}"):
                example = self.serialize_example(row['image_path'], row['mask_path'])
                writer.write(example)
        print(f"✅ Saved {log_title} to {self.tfrecord_path}")