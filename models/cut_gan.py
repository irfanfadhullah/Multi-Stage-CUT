import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from models.base import BaseModel
from models.discriminator import Discriminator
from models.generator import Generator
from models.mlp_projector import Projector_Head


class ContrastiveModel(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)
        self.model_names = ['D_Y', 'G', 'H']
        self.loss_names = ['G_adv', 'D_Y', 'G', 'NCE']
        self.visual_names = ['X', 'Y', 'Y_fake']
        if self.config["TRAINING_SETTING"]["LAMBDA_Y"] > 0:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['Y_idt']

        self.D_Y = Discriminator().to(self.device)
        self.G = Generator().to(self.device)
        self.H = Projector_Head().to(self.device)

        self.opt_D_Y = optim.Adam(self.D_Y.parameters(), lr=self.config["TRAINING_SETTING"]["LEARNING_RATE"], betas=(0.5, 0.999),)
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.config["TRAINING_SETTING"]["LEARNING_RATE"], betas=(0.5, 0.999),)
        self.opt_H = optim.Adam(self.H.parameters(), lr=self.config["TRAINING_SETTING"]["LEARNING_RATE"], betas=(0.5, 0.999),)

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

        if self.config["TRAINING_SETTING"]["LOAD_MODEL"]:
            self.load_networks(self.config["TRAINING_SETTING"]["EPOCH"])

        lambda_lr = lambda epoch: 1.0 - max(0, epoch - self.config["TRAINING_SETTING"]["NUM_EPOCHS"] / 2) / (self.config["TRAINING_SETTING"]["NUM_EPOCHS"] / 2)
        self.scheduler_disc = lr_scheduler.LambdaLR(self.opt_D_Y, lr_lambda=lambda_lr)
        self.scheduler_gen = lr_scheduler.LambdaLR(self.opt_G, lr_lambda=lambda_lr)
        self.scheduler_mlp = lr_scheduler.LambdaLR(self.opt_H, lr_lambda=lambda_lr)

    def set_input(self, input):
        self.X, self.Y = input

    def forward(self):
        self.Y = self.Y.to(self.device)
        self.X = self.X.to(self.device)
        self.Y_fake = self.G(self.X)
        if self.config["TRAINING_SETTING"]["LAMBDA_Y"] > 0:
            self.Y_idt = self.G(self.Y)
    
    def inference(self, X):
        self.eval()
        with torch.no_grad():
            X = X.to(self.device)
            Y_fake = self.G(X)
        return Y_fake

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.D_Y, True)
        self.opt_D_Y.zero_grad()
        self.loss_D_Y = self.compute_D_loss()
        self.loss_D_Y.backward()
        self.opt_D_Y.step()

        # update G and H
        self.set_requires_grad(self.D_Y, False)
        self.opt_G.zero_grad()
        self.opt_H.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.opt_G.step()
        self.opt_H.step()

    def scheduler_step(self):
        self.scheduler_disc.step()
        self.scheduler_gen.step()
        self.scheduler_mlp.step()

    def compute_D_loss(self):
        # Fake
        fake = self.Y_fake.detach()
        pred_fake = self.D_Y(fake)
        self.loss_D_fake = self.mse(pred_fake, torch.zeros_like(pred_fake))
        # Real
        self.pred_real = self.D_Y(self.Y)
        self.loss_D_real = self.mse(self.pred_real, torch.ones_like(self.pred_real))

        self.loss_D_Y = (self.loss_D_fake + self.loss_D_real) / 2
        return self.loss_D_Y

    def compute_G_loss(self):
        fake = self.Y_fake
        pred_fake = self.D_Y(fake)
        self.loss_G_adv = self.mse(pred_fake, torch.ones_like(pred_fake))

        self.loss_NCE = self.calculate_NCE_loss(self.X, self.Y_fake)
        if self.config["TRAINING_SETTING"]["LAMBDA_Y"] > 0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.Y, self.Y_idt)
            self.loss_NCE = (self.loss_NCE + self.loss_NCE_Y) * 0.5

        self.loss_G = self.loss_G_adv + self.loss_NCE
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        feat_q, patch_ids_q = self.G(tgt, encode_only=True)
        feat_k, _ = self.G(src, encode_only=True, patch_ids=patch_ids_q)
        feat_k_pool = self.H(feat_k)
        feat_q_pool = self.H(feat_q)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.patch_nce_loss(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / 5

    def patch_nce_loss(self, feat_q, feat_k):
        feat_k = feat_k.detach()
        out = torch.mm(feat_q, feat_k.transpose(1, 0)) / 0.07
        loss = self.cross_entropy_loss(out, torch.arange(0, out.size(0), dtype=torch.long, device=self.device))
        return loss
    

class MultiStageGAN(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)
        self.model_names = ['G1', 'G2', 'G3', 'D1', 'D2', 'D3', 'H']
        self.loss_names = ['G1_adv', 'G2_adv', 'G3_adv', 
                          'D1', 'D2', 'D3', 
                          'NCE1', 'NCE2', 'NCE3']
        
        self.visual_names = ['A', 'A_mask', 'A_mask_fake',
                           'B_mask', 'B_mask_fake', 
                           'B', 'B_fake']

        # Initialize generators
        self.G1 = Generator().to(self.device)  # A -> A_mask
        self.G2 = Generator().to(self.device)  # A_mask -> B_mask
        self.G3 = Generator().to(self.device)  # B_mask -> B

        # Initialize discriminators
        self.D1 = Discriminator().to(self.device)  # A_mask discriminator
        self.D2 = Discriminator().to(self.device)  # B_mask discriminator
        self.D3 = Discriminator().to(self.device)  # B discriminator

        # Initialize projector head for contrastive learning
        self.H = Projector_Head().to(self.device)

        # Setup optimizers
        self.setup_optimizers()
        
        # Loss functions
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

        if self.config["TRAINING_SETTING"]["LOAD_MODEL"]:
            self.load_networks(self.config["TRAINING_SETTING"]["EPOCH"])

    def setup_optimizers(self):
        lr = self.config["TRAINING_SETTING"]["LEARNING_RATE"]
        beta = (0.5, 0.999)
        
        # Generator optimizers
        self.opt_G1 = optim.Adam(self.G1.parameters(), lr=lr, betas=beta)
        self.opt_G2 = optim.Adam(self.G2.parameters(), lr=lr, betas=beta)
        self.opt_G3 = optim.Adam(self.G3.parameters(), lr=lr, betas=beta)
        
        # Discriminator optimizers
        self.opt_D1 = optim.Adam(self.D1.parameters(), lr=lr, betas=beta)
        self.opt_D2 = optim.Adam(self.D2.parameters(), lr=lr, betas=beta)
        self.opt_D3 = optim.Adam(self.D3.parameters(), lr=lr, betas=beta)
        
        # Projector optimizer
        self.opt_H = optim.Adam(self.H.parameters(), lr=lr, betas=beta)

    def set_input(self, input):
        self.A, self.A_mask, self.B_mask, self.B = input

    def forward(self):
        # Move inputs to device
        self.A = self.A.to(self.device)
        self.A_mask = self.A_mask.to(self.device)
        self.B_mask = self.B_mask.to(self.device)
        self.B = self.B.to(self.device)

        # Generate fake images
        self.A_mask_fake = self.G1(self.A)
        self.B_mask_fake = self.G2(self.A_mask_fake)
        self.B_fake = self.G3(self.B_mask_fake)

    def optimize_parameters(self):
        self.forward()
        
        # Update D1
        self.set_requires_grad([self.D1, self.D2, self.D3], True)
        self.opt_D1.zero_grad()
        self.loss_D1 = self.compute_D_loss(self.D1, self.A_mask_fake, self.A_mask)
        self.loss_D1.backward(retain_graph=True)
        self.opt_D1.step()

        # Update D2
        self.opt_D2.zero_grad()
        self.loss_D2 = self.compute_D_loss(self.D2, self.B_mask_fake, self.B_mask)
        self.loss_D2.backward(retain_graph=True)
        self.opt_D2.step()

        # Update D3
        self.opt_D3.zero_grad()
        self.loss_D3 = self.compute_D_loss(self.D3, self.B_fake, self.B)
        self.loss_D3.backward(retain_graph=True)
        self.opt_D3.step()

        # Update Generators and Projector
        self.set_requires_grad([self.D1, self.D2, self.D3], False)
        self.opt_G1.zero_grad()
        self.opt_G2.zero_grad()
        self.opt_G3.zero_grad()
        self.opt_H.zero_grad()
        
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        
        self.opt_G1.step()
        self.opt_G2.step()
        self.opt_G3.step()
        self.opt_H.step()

    def compute_D_loss(self, D, fake, real):
        pred_fake = D(fake.detach())
        loss_D_fake = self.mse(pred_fake, torch.zeros_like(pred_fake))
        
        pred_real = D(real)
        loss_D_real = self.mse(pred_real, torch.ones_like(pred_real))
        
        return (loss_D_fake + loss_D_real) * 0.5

    def compute_G_loss(self):
        # First stage losses
        self.loss_G1_adv = self.mse(self.D1(self.A_mask_fake), 
                                   torch.ones_like(self.D1(self.A_mask_fake)))
        self.loss_NCE1 = self.calculate_NCE_loss(self.A, self.A_mask_fake)

        # Second stage losses
        self.loss_G2_adv = self.mse(self.D2(self.B_mask_fake), 
                                   torch.ones_like(self.D2(self.B_mask_fake)))
        self.loss_NCE2 = self.calculate_NCE_loss(self.A_mask_fake, self.B_mask_fake)

        # Third stage losses
        self.loss_G3_adv = self.mse(self.D3(self.B_fake), 
                                   torch.ones_like(self.D3(self.B_fake)))
        self.loss_NCE3 = self.calculate_NCE_loss(self.B_mask_fake, self.B_fake)

        return (self.loss_G1_adv + self.loss_G2_adv + self.loss_G3_adv + 
                self.loss_NCE1 + self.loss_NCE2 + self.loss_NCE3)

    def calculate_NCE_loss(self, src, tgt):
        feat_q, patch_ids_q = self.G1(tgt, encode_only=True)
        feat_k, _ = self.G1(src, encode_only=True, patch_ids=patch_ids_q)
        feat_k_pool = self.H(feat_k)
        feat_q_pool = self.H(feat_q)

        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            loss = self.patch_nce_loss(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / 5