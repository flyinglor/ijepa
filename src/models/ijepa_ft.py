import torch
import torch.nn as nn

import src.models.vision_transformer as vit

'''
Finetune IJEPA
'''
class IJEPA_FT(nn.Module):
    #take pretrained model path, number of classes, learning rate, weight decay, and drop path as input
    def __init__(self, 
        encoder=None, 
        num_classes=3, 
        # lr=1e-3, 
        # weight_decay=0, 
        # drop_path=0.1, 
        patch_size=16, 
        model_name='vit_base',
        crop_size=224,
        in_chans=3):

        super().__init__()
        # self.save_hyperparameters()

        # #set parameters
        # self.lr = lr
        # self.weight_decay = weight_decay
        # self.drop_path = drop_path

        if encoder:
            self.encoder = encoder 
        else: #TODO: init encoder
            encoder = vit.__dict__[model_name](
                img_size=[crop_size],
                patch_size=patch_size,
                in_chans=in_chans)

        #TODO: check layer dropout
        # self.encoder.layer_dropout = self.drop_path

################ 3 fc
        self.linear1 = nn.Linear(768, 512)
        self.ln1 = nn.LayerNorm(512, eps=1e-6)
        self.relu1 = nn.ReLU()
        
        self.linear2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512, eps=1e-6)
        self.relu2 = nn.ReLU()
        
        self.avg_pool = nn.AvgPool1d(kernel_size=512, stride=1)
        self.fc = nn.Linear(512, 3)

################# 2 fc
        # self.linear1 = nn.Linear(768, 768)
        # self.ln1 = nn.LayerNorm(768, eps=1e-6)
        # self.relu1 = nn.ReLU()
        
        # self.avg_pool = nn.AvgPool1d(kernel_size=512, stride=1)
        # self.fc = nn.Linear(768, 3)

 ################ 1 fc     
        # self.average_pool = nn.AvgPool1d(kernel_size=512, stride=1)
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(768, eps=1e-6),
        #     nn.Linear(768, num_classes),
        # )
###########
        #define loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):

        ############## 1 fc
        # x = self.encoder(x)
        # # print(x.shape)
        # x = x.transpose(1, 2)
        # # print(x.shape)
        # x = self.average_pool(x) #conduct average pool like in paper
        # x = x.squeeze(-1)
        # x = self.mlp_head(x) #pass through mlp head


        ############### 2 fc
        # x = self.encoder(x)
        # x = self.linear1(x)  # [batch_size, seq_len, hidden_dim]
        # x = self.ln1(x)  # [batch_size, seq_len, hidden_dim]
        # x = self.relu1(x)  # [batch_size, seq_len, hidden_dim]
        # # Transpose for AvgPool1d: [batch_size, hidden_dim, seq_len]
        # x = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
        # # print(x.shape)
        # x = self.avg_pool(x)  # [batch_size, hidden_dim, 1]  
        # # Remove the singleton dimension: [batch_size, hidden_dim]
        # x = x.squeeze(-1)
        # x = self.fc(x)  # [batch_size, output_dim]



        ############## 3fc
        x = self.encoder(x)
        x = self.linear1(x)  # [batch_size, seq_len, hidden_dim]
        x = self.ln1(x)  # [batch_size, seq_len, hidden_dim]
        x = self.relu1(x)  # [batch_size, seq_len, hidden_dim]
        x = self.linear2(x)  # [batch_size, seq_len, hidden_dim]
        x = self.ln2(x)  # [batch_size, seq_len, hidden_dim]
        x = self.relu2(x)  # [batch_size, seq_len, hidden_dim]
        # Transpose for AvgPool1d: [batch_size, hidden_dim, seq_len]
        x = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
        # print(x.shape)
        x = self.avg_pool(x)  # [batch_size, hidden_dim, 1]  
        # Remove the singleton dimension: [batch_size, hidden_dim]
        x = x.squeeze(-1)
        x = self.fc(x)  # [batch_size, output_dim]


        return x
    
