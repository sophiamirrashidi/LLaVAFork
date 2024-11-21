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

    def linear(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
 
    def forward(self, img_tkns, lang_tkns, d_mode):
        # TODO fix the device stuff
        device = 'cuda'  
        loss_function = nn.BCELoss()

        if isinstance(img_tkns, list):
            assert len(img_tkns) == 1, 'img tokens is not a list of length 1'
            img_tkns = img_tkns[0]
        else:  
            print(f'len_img_tkns: {len(img_tkns)}, \n img_tkns: img_tkns')

        img_tkns = img_tkns.view(-1, 5120).to(device)
        img_tkns = img_tkns[:1000]  # Reduce to a smaller subset

        if d_mode:
            img_pred = self.linear(img_tkns.detach())
            img_label = torch.full((img_tkns.size(0), 1), 1, dtype=torch.bfloat16, device=device) # label 1 for images 
            img_loss = loss_function(img_pred, img_label)
            img_correct_count = torch.eq(torch.ge(img_pred, 0.5).float(), img_label).sum().item()
            img_accuracy = img_correct_count / img_tkns.size(0) * 100

            lang_tkns = torch.cat(lang_tkns, dim=0) # batching the language tokens
            lang_tkns = lang_tkns[:1000]  

            print('printing tensor size from inside disc forward pass \n')
            print(f"img_tkn_list shape: {img_tkns.shape}, dtype: {img_tkns.dtype}")
            print(f"lang_tkn_list shape: {lang_tkns.shape}, dtype: {lang_tkns.dtype}")
            # print((torch.cuda.memory_summary()))

            lang_pred = self.linear(lang_tkns.detach())
            lang_label = torch.full((lang_pred.size(0), 1), 0, dtype=torch.bfloat16, device=device)

            lang_loss = loss_function(lang_pred, lang_label)
            lang_correct_count = torch.eq(torch.ge(lang_pred, 0.5).float(), lang_label).sum().item()
            lang_accuracy = lang_correct_count / lang_pred.size(0) * 100

            print(f"Image Accuracy: {img_accuracy:.2f}%")
            print(f"Language Accuracy: {lang_accuracy:.2f}%")

            loss = img_loss + lang_loss
            return loss
        
        else:
            img_pred = self.linear(img_tkns)
            img_with_lang_label_loss = loss_function(img_pred, torch.full((img_tkns.size(0), 1), 0, dtype=torch.bfloat16, device=device))
            return img_with_lang_label_loss