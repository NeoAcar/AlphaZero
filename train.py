import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import functions as f
from resnet import ResNet
from dataset import ChessDataset
from torch.utils.data import DataLoader
from tqdm import tqdm # type: ignore
import json
from torch.utils.tensorboard import SummaryWriter


with open('train_config.json') as file:
    args = json.load(file)
class Train:
    def __init__(self: 'Train') -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResNet().to(self.device)

        self.args = args
        self.max_steps = self.args['max_steps']
        self.lr = self.args['learning_rate']
        self.l2_weight = self.args['l2_weight']
        self.log_step = self.args['log_step']
        self.epochs = self.args['epochs']
        self.batch_size = self.args['batch_size']
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.l2_weight)
        self.games_path = self.args['games_path']
        self.evals_path = self.args['evals_path']

        self.max_games = self.args['max_games']

    def data_preparation(self: 'Train') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        games = f.load_pgn(self.games_path, self.max_games)
        boards,results,moves = f.create_nn_input(games)
        evals = np.load(self.evals_path)
        evals = np.array(evals,dtype=np.float32).reshape(-1,1)
        evals = torch.tensor(evals,dtype=torch.float32)
        return boards,evals,moves

    def train(self: 'Train', boards: torch.Tensor, evals: torch.Tensor, moves: torch.Tensor) -> None:
        writer = SummaryWriter(log_dir='./logs',flush_secs=1)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        checkpoint_dir = './checkpoints'

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Data preparation
        train_dataset = ChessDataset(boards,evals,moves)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        total_step = len(train_loader)
        step = total_step//self.log_step

        # Model, loss functions, optimizer
        criterion_mse = nn.MSELoss()
        criterion_ce = nn.CrossEntropyLoss()


        # Track analytics
        train_loss_history = []
        train_mse_loss_history = []
        train_ce_loss_history = []
        train_accuracy_history = []
        iters = 0
        for epoch in range(self.epochs):
            lr = self.optimizer.param_groups[0]['lr']
            self.model.train()
            running_loss = 0.0
            running_mse_loss = 0.0
            running_ce_loss = 0.0
            correct = 0
            total = 1
            itr_loss = 0.0
            itr_ms_loss = 0.0
            itr_ce_loss = 0.0
            for data, labels_mse, labels_ce in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}'):
                data = data.to(self.device)
                labels_mse = labels_mse.to(self.device)  
                labels_ce = labels_ce.to(self.device)  
                self.optimizer.zero_grad()
            
                outputs_mse, outputs_ce = self.model(data)
                loss_mse = criterion_mse(outputs_mse, labels_mse)   
                loss_ce = criterion_ce(outputs_ce, labels_ce) 
                
                loss = loss_mse + loss_ce
                loss.backward()
                self.optimizer.step()
                
                # Track loss
                running_loss += loss.item()
                running_mse_loss += loss_mse.item()
                running_ce_loss += loss_ce.item()
                

                _, predicted = torch.max(outputs_ce.data, 1)
                total += labels_ce.size(0)
                correct += (predicted == labels_ce).sum().item()

                if iters % self.log_step == 0:
                    loss = running_loss - itr_loss
                    itr_loss = running_loss
                    mse_loss = running_mse_loss - itr_ms_loss
                    itr_ms_loss = running_mse_loss
                    ce_loss = running_ce_loss - itr_ce_loss
                    itr_ce_loss = running_ce_loss


                    log = f'iterasyon {iters}, epoch {(epoch+1)}/{self.epochs}, Loss: {loss / step:.4f}, Learning rate: {lr}, MSE Loss: {mse_loss / step:.4f}, CE Loss: {ce_loss / step:.4f}, Accuracy: {100 * correct / total:.2f}%'
                    with open('log.txt', 'a') as f:
                        f.write(log + '\n')
                    writer.add_scalar('Loss', loss / step, iters)
                    writer.add_scalar('MSE Loss', mse_loss / step, iters)
                    writer.add_scalar('CE Loss', ce_loss / step, iters)
                    writer.add_scalar('Accuracy', 100 * correct / total, iters)
                    writer.add_scalar('Learning rate', lr, iters)
                    writer.flush()
                    writer.close()
                iters += 1


            

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100 * correct / total
            epoch_mse = running_mse_loss / len(train_loader)
            epoch_ce = running_ce_loss / len(train_loader)
            train_loss_history.append(epoch_loss)
            train_mse_loss_history.append(epoch_mse)
            train_ce_loss_history.append(epoch_ce)
            train_accuracy_history.append(epoch_accuracy)

            epoch_log = f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Learning rate: {lr}, MSE Loss: {epoch_mse:.4f}, CE Loss: {epoch_ce:.4f}, Accuracy: {epoch_accuracy:.2f}%"
            print(epoch_log)
            if train_loss_history[-1] == min(train_loss_history):
                for file in os.listdir(checkpoint_dir):
                    if file.startswith('model_epoch_'):
                        os.remove(os.path.join(checkpoint_dir, file))
                e = epoch + 1
                print("Best model so far. Saving...")
                checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{e}.pth')
                torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': epoch_loss,
                'mse_loss': epoch_mse,
                'ce_loss': epoch_ce,
                'accuracy': epoch_accuracy,
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
                
            else:
                lr *= 0.3
                checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{e}.pth')
                self.model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
                self.optimizer.load_state_dict(torch.load(checkpoint_path)["optimizer_state_dict"])
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print("Learning rate decreased to", param_group['lr'])
                print("Best model loaded")

            

        analytics_path = os.path.join(checkpoint_dir, 'training_analytics.pth')
        torch.save({
            'train_loss_history': train_loss_history,
        }, analytics_path)
        print(f"Training analytics saved at {analytics_path}")
        print("Training finished!")

    def predict(self, board, move_counter):
        self.model.eval()
        state = f.prepare_input(board, move_counter).unsqueeze(0).to(self.args['device'])
        value = self.model(state)
        return value.cpu().item()
    def main(self):
        boards,evals,moves = self.data_preparation()
        self.train(boards,evals,moves)

    

    
if __name__ == '__main__':
    train = Train()
    train.main()

