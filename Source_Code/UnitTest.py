import os
import unittest

from DataPreparation import DataPreparation
from Model import LegoPartsModelV1
from Trainer import Trainer
from ConfigParser import ConfigParser


class Homework4Test(unittest.TestCase):
    def test_model(self):
        config = ConfigParser(os.path.join(os.getcwd(), "config.yaml"))
        cfg = config.get_config()

        data_preparation = DataPreparation(
            images_root=cfg["meta"]["images_root"],
            parts_csv_path=cfg["meta"]["parts_csv_path"],
            image_size=cfg["meta"].get("image_size", 128),
            train_split=cfg["meta"].get("train_split", 0.8),
            random_seed=cfg["meta"].get("random_seed", 42),
            grayscale=cfg["meta"].get("grayscale", False),
            selected_parts=cfg["meta"].get("selected_parts"),
            duplicate_singletons_for_test=cfg["meta"].get("duplicate_singletons_for_test", True),
        )

        train_data, test_data = data_preparation.get_data()
        sample_dataset = train_data if len(train_data) > 0 else test_data

        image, label = sample_dataset[0]
        image = image.unsqueeze(0)

        model = LegoPartsModelV1(
            input_shape=data_preparation.image_depth,
            hidden_units=cfg["model"]["LegoPartsModelV1"]["hidden_neurons"],
            output_shape=len(data_preparation.classes),
        )

        prediction = model(image)
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.shape[1], len(data_preparation.classes))

    def test_training(self):
        config = ConfigParser(os.path.join(os.getcwd(), "config.yaml"))
        cfg = config.get_config()

        data_preparation = DataPreparation(
            images_root=cfg["meta"]["images_root"],
            parts_csv_path=cfg["meta"]["parts_csv_path"],
            image_size=cfg["meta"].get("image_size", 128),
            train_split=cfg["meta"].get("train_split", 0.8),
            random_seed=cfg["meta"].get("random_seed", 42),
            grayscale=cfg["meta"].get("grayscale", False),
            selected_parts=cfg["meta"].get("selected_parts"),
            duplicate_singletons_for_test=cfg["meta"].get("duplicate_singletons_for_test", True),
        )

        train_data, test_data = data_preparation.get_data()

        model = LegoPartsModelV1(
            input_shape=data_preparation.image_depth,
            hidden_units=cfg["model"]["LegoPartsModelV1"]["hidden_neurons"],
            output_shape=len(data_preparation.classes),
        )

        training_config = cfg["training"]
        training_config["model"] = model
        training_config["training_data"] = train_data
        training_config["testing_data"] = test_data

        trainer = Trainer(**training_config)
        train_metrics = trainer.train()
        test_metrics = trainer.test()

        self.assertIsNotNone(train_metrics)
        self.assertIn("loss", train_metrics)
        self.assertIn("accuracy", train_metrics)

        self.assertIsNotNone(test_metrics)
        self.assertIn("loss", test_metrics)
        self.assertIn("accuracy", test_metrics)


if __name__ == "__main__":
    unittest.main()