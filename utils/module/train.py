import torch
import optuna
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from .loss_function import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class Train:

    def __init__(self, dataset: torch.utils.data.dataset, model: torch.nn.Module, batch_size=1, lr=1e-4, weight_decay=1e-4, alpha=0.01, weight=torch.tensor([1., 1., 1., 1.])):
        self.batch_size = batch_size
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.make_loader(dataset)
        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = Combined_loss(
            alpha=alpha, device=self.device, weight=weight)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.1)
        self.writer = SummaryWriter()
        self.model.to(self.device)

    def make_loader(self, dataset):
        train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=True)

    def train_step(self):
        self.model.train()
        loss_num = 0
        accuray = torch.zeros((1, 4)).to(self.device)
        for i, (images, masks) in enumerate(tqdm(self.train_loader)):
            images = images.to(self.device)
            masks = masks.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_num += loss.item()
            output = torch.argmax(outputs[0], dim=1, keepdim=True)
            for i in range(self.model.n_classes):
                accuray[0][i] += (torch.sum((output == i) &
                                  (masks == i)) / torch.sum(masks == i))
            # accuray += torch.sum(output == masks).item() / (images.size(-2) * images.size(-1))
        loss_num = loss_num / len(self.train_loader)
        accuray = accuray / (len(self.train_loader) * self.batch_size)
        return loss_num, accuray

    def val_step(self):
        self.model.eval()
        accuray = torch.zeros((1, 4)).to(self.device)
        with torch.no_grad():
            for _, (images, masks) in enumerate(tqdm(self.val_loader)):
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                output = torch.argmax(outputs[0], dim=1, keepdim=True)
                for i in range(self.model.n_classes):
                    accuray[0][i] += (torch.sum((output == i)
                                      & (masks == i)) / torch.sum(masks == i))
                # accuray += torch.sum(output == mask).item() / (image.size(-2) * image.size(-1))
        accuray = accuray / (len(self.val_loader) * self.batch_size)
        return accuray

    def train(self, epochs, save_path=None):
        best_accuray = 0
        for epoch in range(epochs):
            val_accuray = 0
            loss_num, train_accuray = self.train_step()
            val_accuray = self.val_step()
            self.scheduler.step()
            self.writer.add_scalar('train_loss', loss_num, epoch)
            self.writer.add_scalar('train_accuray', train_accuray[0][1], epoch)
            self.writer.add_scalar('val_accuray', val_accuray[0][1], epoch)
            if val_accuray.mean() > best_accuray:
                best_accuray = val_accuray.mean()
                if save_path is not None:
                    print('save model')
                    self.save_model(
                        save_path + f'\{epoch}-{best_accuray:.2f}.pth')
            print(f'epoch:{epoch},best_accuray:{best_accuray},val_accuray:{val_accuray}',
                  '\n', f'train_accuray:{train_accuray}', f'loss:{loss_num}')

        self.writer.close()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict(self, image: torch.Tensor, model_path=None) -> torch.Tensor:
        """
            input:
                image:torch.Tensor.float -> [B,C,H,W]
                model_path:str -> model path
            return:
                torch.Tensor -> [B,C,H,W]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            outputs = self.model(image)
            output = torch.argmax(outputs[0], dim=1, keepdim=True)
        return output

    def predict_show(self, image: torch.Tensor, model_path: str = None) -> torch.Tensor:
        """
            input:
                image:torch.Tensor.float -> [B,C,H,W]
                model_path:str -> model path
            return:
                torch.Tensor -> [B,C,H,W]
        """
        output = self.predict(image, model_path)
        image = image.int()
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        output = output.cpu()
        mask_unique = torch.unique(output)

        # 创建颜色列表
        colors = ['black', 'red', 'green', 'blue', 'yellow']  # 根据你的需要修改这个列表
        cmap = mcolors.ListedColormap(colors[:len(mask_unique)])
        norm = mcolors.BoundaryNorm(mask_unique, len(mask_unique))

        fig1 = ax[0].imshow(image.permute(1, 2, 0))
        fig2 = ax[1].imshow(output.squeeze(0).squeeze(0), cmap=cmap, norm=norm)
        fig.colorbar(fig1, ax=ax[0])
        fig.colorbar(fig2, ax=ax[1])
        ax[0].set_title('image')
        ax[1].set_title(f'{mask_unique}')
        plt.show()
        return output


class Optuna_train:

    def __init__(self, dataset, model, path=None):
        self.dataset = dataset
        self.model = model
        self.best_accuray = 0
        self.path = path

    def objective(self, trial):
        weight1 = trial.suggest_float('weight1', 0.1, 2.0)
        weight2 = trial.suggest_float('weight2', 0.1, 2.0)
        weight3 = trial.suggest_float('weight3', 0.1, 2.0)
        weight4 = trial.suggest_float('weight4', 0.1, 2.0)
        weight = torch.tensor([weight1, weight2, weight3, weight4])

        trainer = Train(self.dataset, self.model, weight=weight)
        trainer.train(20)
        val_accuracy = trainer.val_step().mean()
        if val_accuracy > self.best_accuray:
            self.best_accuray = val_accuracy
            if self.path:
                print('save model')
                torch.save(trainer.model.state_dict(), self.path +
                           f'/best-{self.best_accuray:.2f}.pth')
        return val_accuracy

    def opt(self, n_trials=10):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        print(study.best_params)
