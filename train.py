from model_utils.feature_columns import Feature_columns
import tensorflow as tf
from config import data_config, model_config
from model.FM import FM


# 设置显存使用按需增长
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def get_feature_columns(data_config):
    # 解析data_coinfig
    categorical_column_list = data_config["categorical_column_list"]
    numeric_bucket_column_list = data_config["numeric_bucket_column_list"]

    # 准备feature_columns
    feature_columns = []

    for feature_col in categorical_column_list + numeric_bucket_column_list:
        feature_encoder = getattr(Feature_columns, feature_col["feature_type"])
        column_name = feature_col.pop("column_name")
        feature_columns.append(feature_encoder(None, column_name=column_name, **feature_col))
    return feature_columns


def get_inputs(data_config):
    categorical_column_list = data_config["categorical_column_list"]
    numeric_bucket_column_list = data_config["numeric_bucket_column_list"]
    categorical_features = [categorical_column["column_name"] for categorical_column in categorical_column_list]
    numeric_features = [numeric_bucket_column["column_name"] for numeric_bucket_column in numeric_bucket_column_list]
    feature2dtype = data_config["feature2dtype"]
    # 准备inputs
    inputs = {
        feature: tf.keras.layers.Input(name=feature, shape=(), dtype=feature2dtype[feature]) for feature in
        numeric_features + categorical_features
    }
    return inputs

def get_datasets(data_config, stage):
    if stage == "train":
        file_source = data_config["train_val_data_source"]
    else:
        file_source = data_config["test_data_source"]
    ds = tf.data.experimental.make_csv_dataset(
        file_pattern=file_source,
        batch_size=data_config["batch_size"],
        label_name=data_config["label_name"],
        na_value="",
        num_epochs=1,
        shuffle=data_config["shuffle"],
        ignore_errors=True)
    return ds

if __name__=="__main__":
    # 根据配置获取数据集
    train_ds = get_datasets(data_config, "train")
    val_ds = get_datasets(data_config, "test")


    # 解析model_config
    latent_feature_dim = model_config["latent_feature_dim"]
    lr = model_config["learn_rate"]
    epochs = model_config["epochs"]
    patience = model_config["patience"]
    model_save_path = model_config["model_save_path"]

    # build model
    inputs = get_inputs(data_config)
    feature_columns = get_feature_columns(data_config)
    model = FM(inputs, feature_columns, latent_feature_dim=latent_feature_dim)
    model.compile_model(lr)
    # train model
    model.train(data=train_ds, val_data=val_ds, epochs=epochs, patience=patience, model_file_path=model_save_path)
