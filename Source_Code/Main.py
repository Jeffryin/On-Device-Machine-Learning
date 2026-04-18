import os
import sys
import logging

from DataPreparation import DataPreparation
from Model import LegoPartsModelV1
from ConfigParser import ConfigParser
from Trainer import Trainer
from Logger import Logger

if os.path.exists("mylog.log"):
    log_file_size = os.path.getsize("mylog.log")
    if log_file_size > 2 * 1024 * 1024:
        os.remove("mylog.log")
        print("The log file was greater than 2MB and has been deleted.")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="mylog.log",
    filemode="a",
)

stdout_logger = logging.getLogger("STDOUT")
sys.stdout = Logger(stdout_logger, logging.DEBUG)

config = ConfigParser(os.path.join(os.getcwd(), "config.yaml"))
cfg = config.get_config()

meta_config = cfg["meta"]
model_config = cfg["model"]
training_config = cfg["training"]

data_preparation = DataPreparation(
    images_root=meta_config["images_root"],
    parts_csv_path=meta_config["parts_csv_path"],
    image_size=meta_config.get("image_size", 128),
    train_split=meta_config.get("train_split", 0.8),
    random_seed=meta_config.get("random_seed", 42),
    grayscale=meta_config.get("grayscale", False),
    selected_parts=meta_config.get("selected_parts"),
    duplicate_singletons_for_test=meta_config.get("duplicate_singletons_for_test", True),
)

train_data, test_data = data_preparation.get_data()

model = LegoPartsModelV1(
    input_shape=data_preparation.image_depth,
    hidden_units=model_config["LegoPartsModelV1"]["hidden_neurons"],
    output_shape=len(data_preparation.classes),
)

training_config["model"] = model
training_config["training_data"] = train_data
training_config["testing_data"] = test_data

trainer = Trainer(**training_config)
train_metrics = trainer.train()
test_metrics = trainer.test()

print(f"Number of classes: {len(data_preparation.classes)}")
print(f"Train metrics: {train_metrics}")
print(f"Test metrics: {test_metrics}")

# Example label lookup
print("\nSample label mapping:")
for idx in list(data_preparation.idx_to_part_num.keys())[:5]:
    print(
        f"Class {idx}: "
        f"part_num={data_preparation.get_part_num_from_index(idx)} | "
        f"name={data_preparation.get_part_name_from_index(idx)}"
    )