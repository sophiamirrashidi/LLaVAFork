import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 25)
        self.fc2 = nn.Linear(25, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        
    def linear(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
 
    def forward(self, img_tkn_list, lang_tkn_list, d_mode):
        device = 'cuda'  
        loss_function = nn.BCELoss()

        if isinstance(img_tkn_list, list):
            assert len(img_tkn_list) == 1, 'img tokens is not a list of length 1'
            img_tkn_list = img_tkn_list[0]
        else:  
            print(f'len_img_tkns: {len(img_tkn_list)}, \n img_tkns: img_tkns')

        assert img_tkn_list.size(-1) == 5120, "Expected input size of 5120 for img_tkns"
        img_tkns = img_tkn_list.view(-1, 5120)

        lang_tkns = torch.cat(lang_tkn_list, dim=0) # batching the language tokens 

        if d_mode:
            img_pred = self.linear(img_tkns.detach())
            # img_label = torch.full((img_tkns.size(0), 1), 1, dtype=torch.bfloat16, device=device) # label 1 for images
            img_label = torch.full((img_tkns.size(0), 1), 0.9, dtype=torch.bfloat16, device=device) + torch.rand((img_tkns.size(0), 1), device=device) * 0.1
            img_loss = loss_function(img_pred, img_label.type(torch.bfloat16))
            img_correct_count = torch.eq(torch.ge(img_pred, 0.5).float(), img_label).sum().item()
            img_accuracy = img_correct_count / img_tkns.size(0) * 100

            lang_pred = self.linear(lang_tkns.detach())
            # lang_label = torch.full((lang_pred.size(0), 1), 0, dtype=torch.bfloat16, device=device)
            lang_label = torch.full((lang_pred.size(0), 1), 0.1, dtype=torch.bfloat16, device=device) + torch.rand((lang_pred.size(0), 1), device=device) * 0.1

            lang_loss = loss_function(lang_pred, lang_label.type(torch.bfloat16))
            lang_correct_count = torch.eq(torch.ge(lang_pred, 0.5).float(), lang_label).sum().item()
            lang_accuracy = lang_correct_count / lang_pred.size(0) * 100

            print(f"Image Accuracy: {img_accuracy:.2f}%")
            print(f"Language Accuracy: {lang_accuracy:.2f}%")

            loss = img_loss + lang_loss
            return loss
        
        else:
            img_pred = self.linear(img_tkns)
            # img_with_lang_label_loss = loss_function(img_pred, torch.full((img_tkns.size(0), 1), 0, dtype=torch.bfloat16, device=device))
            img_label = torch.full((img_pred.size(0), 1), 0.1, dtype=torch.bfloat16, device=device) + torch.rand((img_pred.size(0), 1), device=device) * 0.1
            img_with_lang_label_loss = loss_function(img_pred, img_label.type(torch.bfloat16))
            return img_with_lang_label_loss