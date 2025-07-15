import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import io


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(image, mask):
    def encode_img(img):
        with io.BytesIO() as output:
            img.save(output, format="PNG")
            return output.getvalue()

    image_bytes = encode_img(image)
    mask_bytes = encode_img(mask)

    feature = {
        'image': _bytes_feature(image_bytes),
        'mask': _bytes_feature(mask_bytes),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def write_tfrecord(dataset_df, tfrecord_path, log_title="TFRecord"):
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for idx, row in tqdm(dataset_df.iterrows(), total=len(dataset_df), desc=f"Writing {log_title}"):
            image = Image.open(row['image_path']).convert("L")
            mask = Image.open(row['mask_path']).convert("L")
            example = serialize_example(image, mask)
            writer.write(example)
    print(f"✅ Saved {log_title} to {tfrecord_path}")

