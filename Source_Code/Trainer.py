import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model,
        training_data,
        testing_data,
        optimizer,
        criterion,
        epochs,
        learning_rate,
        batch_size,
        num_workers=0,
    ):
        self.model = model
        self.training_loader = DataLoader(
            training_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        self.testing_loader = DataLoader(
            testing_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.criterion_name = criterion

        self.model.to(self.device)
        self.set_training_config()

    def set_training_config(self):
        if self.optimizer_name == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError("Supported optimizers: SGD, Adam")

        if self.criterion_name == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Use CrossEntropyLoss for multiclass classification")

    def get_training_config(self):
        return {
            "model": self.model,
            "training_loader": self.training_loader,
            "testing_loader": self.testing_loader,
            "optimizer": self.optimizer,
            "criterion": self.criterion,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
        }

    def train(self):
        self.model.train()
        last_avg_loss = 0.0
        last_acc = 0.0

        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.training_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                predicted = logits.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            last_avg_loss = running_loss / max(1, len(self.training_loader))
            last_acc = 100.0 * correct / max(1, total)

            print(
                f"Epoch: {epoch + 1}/{self.epochs} | "
                f"Train Loss: {last_avg_loss:.4f} | "
                f"Train Acc: {last_acc:.2f}%"
            )

        return {"loss": float(last_avg_loss), "accuracy": float(last_acc)}

    def test(self):
        if len(self.testing_loader.dataset) == 0:
            print("Test set is empty. Skipping evaluation.")
            return {"loss": None, "accuracy": None}

        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.inference_mode():
            for images, labels in self.testing_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                running_loss += loss.item()
                predicted = logits.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = running_loss / max(1, len(self.testing_loader))
        acc = 100.0 * correct / max(1, total)

        print(f"Test Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")
        return {"loss": float(avg_loss), "accuracy": float(acc)}