import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 1)

    def linear(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
 
    def forward(self, img_tkns, lang_tkns, d_mode):
        # TODO fix the device stuff
        device = 'cuda'  
        loss_function = nn.BCELoss()

        assert len(img_tkns == 1), 'img tokens is not a list of length 1'

        image_batch = img_tkns[0].view(-1, 5120).to(device)
        img_tkns = image_batch.view(-1, 5120)  # flatten the lists
        img_pred = self.linear(img_tkns)

        if d_mode:
            img_label = torch.full((img_tkns.size(0), 1), 1, dtype=torch.bfloat16, device=device)  # use label 1 for imgs
            img_loss = loss_function(img_pred, img_label)
            
            total_lang_loss = 0

            for lang in lang_tkns:
                lang_pred = self.linear(lang.view(-1, 5120))
                lang_label = torch.full((lang_pred.size(0), 1), 0, dtype=torch.bfloat16, device=device)

                lang_loss = loss_function(lang_pred, lang_label)
                total_lang_loss += lang_loss
                
            img_correct = torch.eq(torch.ge(img_pred, 0.5).float().to(torch.bfloat16), img_label).sum().item()
            lang_correct = torch.eq(torch.ge(lang_pred, 0.5).float().to(torch.bfloat16), lang_label).sum().item()

            img_accuracy = img_correct / img_tkns.size(0) * 100
            lang_accuracy = lang_correct / lang_tkns.size(0) * 100

            print(f"Image Accuracy: {img_accuracy:.2f}%")
            print(f"Language Accuracy: {lang_accuracy:.2f}%")

            loss = img_loss + lang_loss

            return loss
        
        else:
            img_with_lang_label_loss = loss_function(img_pred, torch.full((img_tkns.size(0), 1), 0, dtype=torch.bfloat16, device=device))
            return img_with_lang_label_loss