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

    def run_forward(self, data, d_mode):
        # device = data['image'][0].device
        rank = int(os.environ.get('RANK', -1))
        local_rank = int(os.environ.get('LOCAL_RANK', -1))

        torch.cuda.set_device(local_rank)

        data['lang'] = [lang for lang in data['lang']]
        image_list = [img  for img in data['image']]

        zipped_lists = list(zip(image_list, data["lang"]))

        total_loss = 0
        img_is_correct = 0
        lang_is_correct = 0

        for image, language in zipped_lists:
            if d_mode:
                loss, img, lang = self.forward(image.view(-1, 5120), language.view(-1, 5120), d_mode, device= torch.device(f'cuda:{local_rank}'))
                # loss, img, lang = self.forward(image.view(-1, 5120), language.view(-1, 5120), d_mode)
                total_loss += loss / len(zipped_lists)
                img_is_correct += torch.sum(img)
                lang_is_correct += torch.sum(lang)
            else:
                total_loss += self.forward(image.view(-1, 5120), language.view(-1, 5120), d_mode, device= torch.device(f'cuda:{local_rank}')) / len(zipped_lists)
                # total_loss += self.forward(image.view(-1, 5120), language.view(-1, 5120), d_mode) / len(zipped_lists)


        print("dmode: ", d_mode, "image is correct: ", img_is_correct)
        print("dmode: ", d_mode, "lang is correct: ", lang_is_correct)

        if d_mode:
            return {
                "loss": total_loss,
                "img_is_correct": img_is_correct,
                "lang_is_correct": lang_is_correct,
            }

        return total_loss

    def forward(self, img_tok, lang_tok, d_mode, device):
        loss_function = nn.BCELoss()

        img_tok, lang_tok = img_tok.to(device), lang_tok.to(device)

        img_pred = self.linear(img_tok)
        lang_pred = self.linear(lang_tok)

        if d_mode:
            img_label = torch.full(
                (img_tok.size(0), 1), 1, dtype=torch.bfloat16, device=device
            )
            lang_label = torch.full(
                (lang_tok.size(0), 1), 0, dtype=torch.bfloat16, device=device
            )

            img_loss = loss_function(img_pred, img_label)
            lang_loss = loss_function(lang_pred, lang_label)
            loss = img_loss + lang_loss

            img_pred_binary = torch.ge(img_pred, 0.5).float().to(torch.bfloat16)
            lang_pred_binary = torch.ge(lang_pred, 0.5).float().to(torch.bfloat16)

            img_is_correct = torch.eq(img_pred_binary, img_label).to(device)
            lang_is_correct = torch.eq(lang_pred_binary, lang_label).to(device)
            # img_is_correct = torch.eq(img_pred_binary, img_label)
            # lang_is_correct = torch.eq(lang_pred_binary, lang_label)

            img_total = img_is_correct.size(0)
            lang_total = lang_is_correct.size(0)

            accuracy = (img_is_correct.sum().item() + lang_is_correct.sum().item()) / (img_total + lang_total)
            wandb.log({'disc_acc': accuracy})

            return loss, img_is_correct, lang_is_correct
        else:
            lang_label = torch.full((img_tok.size(0), 1), 0, dtype=torch.bfloat16, device=device)
            lang_label = torch.full((img_tok.size(0), 1), 0, dtype=torch.bfloat16, device=device)
            # lang_label = torch.full(
            #     (img_tok.size(0), 1), 0, dtype=torch.bfloat16, device=torch.device(f'cuda:{local_rank}')
            # )
            img_with_lang_label_loss = loss_function(img_pred, lang_label)
            return img_with_lang_label_loss