import os 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
from sklearn.mixture import GaussianMixture
from EncDec import *

# print('yo')
#  --------------------------------------- MAIN FUNCTIONS ---------------------------------------
def load_dataset(aug_images):
    image_datasets_dict={}

    total_size=len(aug_images)
    train_size=int(0.9*total_size)
    val_size=int(0.05*total_size)
    test_size=total_size-train_size-val_size
    torch.manual_seed(42)
    train_aug_images,val_aug_images,test_aug_images=random_split(aug_images, [train_size, val_size, test_size])
    image_datasets_dict={'train':train_aug_images,"val":val_aug_images,"test":test_aug_images,"all":aug_images}
    return image_datasets_dict
    
  
  
def create_gmm_mapping(aug_images_info,aug_labels,clean_images_info,clean_labels):
    clean_images=[a[2] for a in clean_images_info]
    clean_pca= PCA(n_components=128)
    clean_flattened_data=torch.stack(clean_images).view(len(clean_images),-1).numpy()
    clean_transformed_data=clean_pca.fit_transform(clean_flattened_data)
    clean_transformed_data_dict={a:[] for a in range(0,10)}
    for idx,img in enumerate(clean_transformed_data):
        clean_transformed_data_dict[clean_labels[idx]].append(img)
    for i in clean_transformed_data_dict: clean_transformed_data_dict[i]=np.array(clean_transformed_data_dict[i])
        
    aug_images=[a[2] for a in aug_images_info]
    aug_pca= PCA(n_components=128)
    aug_flattened_data=torch.stack(aug_images).view(len(aug_images),-1).numpy()
    aug_transformed_data=aug_pca.fit_transform(aug_flattened_data)
    aug_transformed_data_dict={a:[] for a in range(0,10)}
    for idx,img in enumerate(aug_transformed_data):
        aug_transformed_data_dict[aug_labels[idx]].append(img)
    for i in aug_transformed_data_dict: aug_transformed_data_dict[i]=np.array(aug_transformed_data_dict[i])
    
    gmms=[]
    for i in range(0,10): 
        gmm=GaussianMixture(n_components=125)
        gmm.fit(clean_transformed_data_dict[i])
        gmms.append(gmm)
        
    
    
    mapping={}
    dict_id_of_best_clean_img_of_each_cluster={}
    for i in range(0,10): 
        curr_gmm=gmms[i]
        probs=curr_gmm.predict_proba(clean_transformed_data_dict[i])
        max_indices = np.argmax(probs, axis=0) # for each component we now have the best clean image
        for idx,a in enumerate(max_indices): # a is the idx of best clean image
            dict_id_of_best_clean_img_of_each_cluster[(i,idx)]=a
            # idx is the cluster no. here 
            # a is the best image idx
            
            
    for i in range(0,10): 
        curr_gmm=gmms[i]
        probs=curr_gmm.predict_proba(aug_transformed_data_dict[i])
        max_indices = np.argmax(probs, axis=1) # for each aug image we now have the component it should belong to  and also the corres clean image 
        for idx,a in enumerate(max_indices): # a is the idx of best clean image
            mapping[(i,idx)]=dict_id_of_best_clean_img_of_each_cluster[(i,a)] # idx should be id of image having index = idx in aug_reduced dataset[i]
            # idx is the image no. here 
            # a is the best cluster idx 
    return mapping




# ------------------------------- AUTO-ENCODER FUNCTIONS -----------------------------------------
def ParameterSelector(E, D):
        return iter(list(E.parameters()) + list(D.parameters()))
    
    
def calculate_ssim_batch(noisy_images,clean_images):
    ssim_values=[]
    for i in range(0,BATCH_SIZE):
        img1_tensor=noisy_images[i]
        img2_tensor=clean_images[i]
        img1=img1_tensor.detach().cpu().numpy().transpose(1, 2, 0).squeeze(2)
        img2=img2_tensor.detach().cpu().numpy().transpose(1, 2, 0).squeeze(2)
        ssim_val=ssim(img1,img2, multichannel=False,data_range=1.0)
        ssim_values.append(ssim_val)
    return ssim_values

def calculate_ssim_single(noisy_image,clean_image):
    img1_tensor=noisy_image[0]
    img2_tensor=clean_image[0]
    img1=img1_tensor.detach().cpu().numpy().transpose(1, 2, 0).squeeze(2)
    img2=img2_tensor.detach().cpu().numpy().transpose(1, 2, 0).squeeze(2)
    ssim_val=ssim(img1,img2, multichannel=False,data_range=1.0)
    return ssim_val
    
    
    
def plot_tsne(train_dataloader,device,encoder,decoder,epoch,type='DAE'):
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        
        encoded_data = []
        labels=[]
        for data in train_dataloader:
            noisy_inputs,clean_inputs,label_list = data
            noisy_inputs.to(device)
            clean_inputs.to(device)
            labels.extend(label_list)
            
            if type=='DAE':
                encoded= encoder(noisy_inputs)
                encoded_data.append(encoded.view(encoded.size(0), -1).cpu())
            else: 
                encoded= encoder(noisy_inputs)                  
                mu, logvar = encoded[:, :encoder.latent_dim], encoded[:,encoder.latent_dim:] 
                encoded_data.append(mu.view(encoded.size(0), -1).cpu())
                
        encoded_features = torch.cat(encoded_data, dim=0)
    
    tsne = TSNE(n_components=3, random_state=42)
    tsne_results = tsne.fit_transform(encoded_features)
    
    # Generate a color map based on labels
    unique_labels = np.unique(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    # Plot t-SNE visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, label in enumerate(unique_labels):
            idx = labels == label
            ax.scatter(tsne_results[idx, 0], tsne_results[idx, 1], tsne_results[idx, 2], color=colors[i], label=str(label))

    # ax.scatter(tsne_results[:,0], tsne_results[:,1], tsne_results[:,2], c='b', marker='o',cmap='tab10')
    # ax.scatter(tsne_results[:,0], tsne_results[:,1], tsne_results[:,2], marker='o',cmap='tab10')
    ax.set_title(f"t-SNE Visualization after {epoch + 1} epochs")
    if type=='DAE': plt.savefig(f'AE_epoch_{epoch+1}.png')
    else: plt.savefig(f'VAE_epoch_{epoch+1}.png')


def save_checkpoints(epoch,loss,optimizer,encoder,decoder,checkpoint_dir1,checkpoint_dir2):
    torch.save({
            'epoch': epoch + 1,
            'model_state_dict': encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,}, checkpoint_dir1)
            
    torch.save({
    'epoch': epoch + 1,
    'model_state_dict': decoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,}, checkpoint_dir2)
    


# -----------------------------GIVEN FUNCTIONS----------------------------------------------
def peak_signal_to_noise_ratio(img1, img2):
    if img1.shape[0] != 1: raise Exception("Image of shape [1,H,W] required.")
    img1, img2 = img1.to(torch.float64), img2.to(torch.float64)
    mse = img1.sub(img2).pow(2).mean()
    if mse == 0: return float("inf")
    else: return 20 * torch.log10(255.0/torch.sqrt(mse)).item()
#--------------------------------------------------------------------------------------------
    






# DATA PREPROCESSING AND LOADING   
aug_data_directory="./Data/aug/"
clean_data_directory="./Data/clean/"
data_transforms = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize images to 28x28
    transforms.Grayscale(),  # Convert images to grayscale
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])
aug_images=[]
aug_labels=[]
aug_images_dict={a:[] for a in range(0,10)}
aug_list_to_dict={}
clean_images=[]
clean_labels=[]
clean_images_dict={a:[] for a in range(0,10)}
clean_list_to_dict={}
image_datasets_dict={}

cnt=0
for img_name in os.listdir(clean_data_directory):
    img_path=os.path.join(clean_data_directory,img_name)
    img=data_transforms(TF.to_pil_image(read_image(img_path)))
    label=int(img_name.split('_')[-1].split('.')[0])  # Extract label from filename
    id=int(img_name.split('_')[1])  # Extract id from filename
    clean_images_dict[label].append(img)
    clean_images.append([id,label,img,cnt])
    clean_labels.append(label)
    clean_list_to_dict[cnt]=(label,len(clean_images_dict[label])-1)
    cnt+=1
    
    
cnt=0
for img_name in os.listdir(aug_data_directory):
    img_path=os.path.join(aug_data_directory,img_name)
    img=data_transforms(TF.to_pil_image(read_image(img_path)))
    label=int(img_name.split('_')[-1].split('.')[0])  # Extract label from filename
    id=int(img_name.split('_')[1])  # Extract id from filename
    aug_images_dict[label].append(img)
    aug_images.append([id,label,img,cnt])
    aug_labels.append(label)
    aug_list_to_dict[cnt]=(label,len(aug_images_dict[label])-1)
    cnt+=1
    
image_datasets_dict=load_dataset(aug_images)
mapping_gmm=create_gmm_mapping(aug_images,aug_labels,clean_images,clean_labels)    



# CUSTOM DATASET PREPARATION
class AlteredMNIST:
    """
    dataset description:
    
    X_I_L.png
    X: {aug=[augmented], clean=[clean]}
    I: {Index range(0,60000)}
    L: {Labels range(10)}
    
    Write code to load Dataset
    """
    def __init__(self, split:str="all") -> None:
        super().__init__()
        if split not in ["train", "test", "val","all"]:
            raise Exception("Data split must be in [train, test, val]")
        global mapping_gmm,image_datasets_dict,clean_images_dict,aug_list_to_dict
        self.datasplit = split
        self.noisy_images=image_datasets_dict[split]
        self.clean_images_dict=clean_images_dict
        self.mapping=mapping_gmm

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        id,label,noisy_image,index=self.noisy_images[idx]
        dict_label,dict_idx=aug_list_to_dict[index]
        clean_image=self.clean_images_dict[label][self.mapping[(dict_label,dict_idx)]]
        return (noisy_image,clean_image,label)



# ENCODER-DECODER BLOCKS 
# resnet block
class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1,ker1_size=3,ker2_size=3):
        super(BasicBlock, self).__init__()
        
        pad1=(ker1_size-1)//2
        pad2=(ker2_size-1)//2
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=ker1_size, stride=stride, padding=pad1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=ker2_size, stride=1, padding=pad2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out




class Encoder(nn.Module):
    """
    Write code for Encoder ( Logits/embeddings shape must be [batch_size,channel,height,width] )
    """
    def __init__(self,Type='DAE') -> None:
        super(Encoder, self).__init__()
        
        
        self.in_channels = 1
        self.Type=Type
        if Type=='DAE':
            self.layer1 = self._make_layer(16,1,2) # (1x28x28) -> (8X14X14)
            self.layer2 = self._make_layer(32,1,2) # (8X14X14) -> (16X7X7)
            self.layer3 = self._make_layer(64,1,2) # (16X7X7) -> (32X3X3)
            self.layer4 = self._make_layer(128,1,2) # (32X3X3) -> (64X1X1)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 64X1 vector
        else: 
            self.layer1 = self._make_layer(16,1,2) # (1x28x28) -> (8X14X14)
            self.layer2 = self._make_layer(32,1,2) # (8X14X14) -> (16X7X7)
            self.layer3 = self._make_layer(64,1,2) # (16X7X7) -> (32X3X3)
            self.layer4 = self._make_layer(128,1,2) # (32X3X3) -> (64X1X1)
            self.latent_dim=32
            self.latent= nn.Linear(128, 2 * self.latent_dim)  
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 64X1 vector
        
        
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride]*num_blocks
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        if self.Type=='DAE':
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out=self.avgpool(out)
            return out
        else: 
            out = self.layer1(x)                        
            out = self.layer2(out)            
            out = self.layer3(out)            
            out = self.layer4(out)            
            out=self.avgpool(out)
            out=out.view(out.size(0),-1)
            out= self.latent(out)
            return out
        
   

class Decoder(nn.Module):
    """
    Write code for decoder here ( Output image shape must be same as Input image shape i.e. [batch_size,1,28,28] )
    """
    def __init__(self,Type='DAE') -> None:
        super(Decoder, self).__init__()
        self.in_channels = 1
        self.Type=Type
        if Type=='DAE':
            self.conv_trans0=nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1) #(64x1x1) -> (64)
            self.bn0=nn.BatchNorm2d(128)
            self.conv_trans1=nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1) #(64x1x1) -> (64)
            self.bn1=nn.BatchNorm2d(64)
            self.relu=nn.ReLU()
            self.conv_trans2=nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
            self.bn2=nn.BatchNorm2d(32)
            self.conv_trans3=nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1)
            self.bn3=nn.BatchNorm2d(16)
            self.conv_trans4=nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1)
            self.sigmoid=nn.Sigmoid()
            
        else:
            self.latent_dim=32
            self.latent=nn.Linear(self.latent_dim, 128)
            self.conv_trans0=nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1) #(64x1x1) -> (64)
            self.bn0=nn.BatchNorm2d(128)
            self.conv_trans1=nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1) #(64x1x1) -> (64)
            self.bn1=nn.BatchNorm2d(64)
            self.relu=nn.ReLU()
            self.conv_trans2=nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
            self.bn2=nn.BatchNorm2d(32)
            self.conv_trans3=nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1)
            self.bn3=nn.BatchNorm2d(16)
            self.conv_trans4=nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1)
            self.sigmoid=nn.Sigmoid()
        
        
    def forward(self, x):
        if self.Type=='DAE':
            out = self.relu(self.bn0(self.conv_trans0(x,output_size=(64,128,2,2))))
            out = self.relu(self.bn1(self.conv_trans1(out,output_size=(64,64,4,4))))
            out = self.relu(self.bn2(self.conv_trans2(out,output_size=(64,32,7,7))))
            out = self.relu(self.bn3(self.conv_trans3(out,output_size=(64,16,14,14))))
            out = self.sigmoid(self.conv_trans4(out,output_size=(64,1,28,28)))
            return out
        else: 
            out = self.latent(x)
            out=out.view(out.size(0),out.size(1),1,1)
            out = self.relu(self.bn0(self.conv_trans0(out,output_size=(64,128,2,2))))
            out = self.relu(self.bn1(self.conv_trans1(out,output_size=(64,64,4,4))))
            out = self.relu(self.bn2(self.conv_trans2(out,output_size=(64,32,7,7))))
            out = self.relu(self.bn3(self.conv_trans3(out,output_size=(64,16,14,14))))
            out = self.sigmoid(self.conv_trans4(out,output_size=(64,1,28,28)))
            return out
       





# DENOISING AUTOENCODER 
checkpoint_dir='./checkpoints/DAE/'    
os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
checkpoint_dir_ae1='./checkpoints/DAE/encoder.pt'    
checkpoint_dir_ae2='./checkpoints/DAE/decoder.pt'

class AELossFn:
    """
    Loss function for AutoEncoder Training Paradigm
    """
    def __init__(self):
        self.criterion = nn.MSELoss()





class AETrainer:
    """
    Write code for training AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as AE_epoch_{}.png
    """
    def __init__(self,dataloader,encoder,decoder,loss_fn,optimizer,gpu='F'):
        self.dataloader=dataloader
        self.encoder=encoder
        self.decoder=decoder
        self.loss_fn=loss_fn.criterion
        self.optimizer=optimizer 
        self.gpu=gpu
        self.train()
        
    def train(self):    
        # train_dataloader=DataLoader(dataset=AlteredMNIST(),batch_size=BATCH_SIZE,shuffle=True,drop_last=True,pin_memory=True)
        train_dataloader=self.dataloader
        # Training loop
        device = torch.device("cuda:0") if self.gpu == "T" else torch.device("cpu")
        for epoch in range(EPOCH):
            
            train_ssim_data=[]
            train_ssim_values=[]
            train_loss=0
            self.encoder.train()
            self.decoder.train()
            
            for idx,data in enumerate(train_dataloader):      
                minibatch=idx+1
                noisy_inputs,clean_inputs,labels = data
                noisy_inputs.to(device)
                clean_inputs.to(device)
                                
                encoded= self.encoder(noisy_inputs)                                
                decoded=self.decoder(encoded)                               
                loss = self.loss_fn(decoded, clean_inputs)                
                train_loss+=loss.item()                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                batch_ssim = calculate_ssim_batch(decoded,clean_inputs)
                train_ssim_values.extend(batch_ssim)
                train_ssim_data.extend(batch_ssim)
                
                # print ssim after every 10th minibatch
                if (minibatch) % 10 == 0:
                    with torch.no_grad():
                        avg_ssim = np.mean(train_ssim_values)
                        similarity=avg_ssim
                        print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch+1,minibatch,train_loss/minibatch,similarity))
                        train_ssim_values = []         
            train_similarity=np.mean(train_ssim_data)
            
            # save checkpoint and print per epoch info
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch+1,train_loss/len(train_dataloader),train_similarity))
            
                        
            # Compute embeddings after every 10 epochs and plot t-SNE
            if (epoch + 1) % 10 == 0:
                plot_tsne(train_dataloader,device,self.encoder,self.decoder,epoch)  


        save_checkpoints(epoch,loss,self.optimizer,self.encoder,self.decoder,checkpoint_dir_ae1,checkpoint_dir_ae2) 

class AE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    def __init__(self,gpu):
        self.gpu=gpu

    def from_path(self,sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        checkpoint1=torch.load(checkpoint_dir_ae1)
        checkpoint2=torch.load(checkpoint_dir_ae2)
        encoder=Encoder()
        decoder=Decoder()
        encoder.load_state_dict(checkpoint1['model_state_dict'])
        decoder.load_state_dict(checkpoint2['model_state_dict'])
        encoder.eval()
        decoder.eval()
    
        device= torch.device("cuda:0") if self.gpu == "T" else torch.device("cpu")
        
        inputs=sample,original
        noisy_img=data_transforms(TF.to_pil_image(read_image(sample)))
        clean_img=data_transforms(TF.to_pil_image(read_image(original)))
        
        noisy_img.to(device)
        clean_img.to(device)
        
        noisy_img=noisy_img.view(1,noisy_img.size(0),noisy_img.size(1),noisy_img.size(2))
        clean_img=clean_img.view(1,clean_img.size(0),clean_img.size(1),clean_img.size(2))
        
        encoded= encoder(noisy_img)                                
        decoded=decoder(encoded)                                    
        if type=='SSIM': score = calculate_ssim_single(decoded,clean_img)
        else: score=peak_signal_to_noise_ratio(decoded,clean_img)
        return score







# VARIATIONAL AUTOENCODER 
checkpoint_dir='./checkpoints/VAE/'    
os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
checkpoint_dir_vae1='./checkpoints/VAE/encoder.pt'    
checkpoint_dir_vae2='./checkpoints/VAE/decoder.pt'


class VAELossFn:
    """
    Loss function for Variational AutoEncoder Training Paradigm
    """
    def __init__(self):
        self.mse_loss = nn.MSELoss()
               
        
    def vae_loss(self,reconstructed_x, x, mu, logvar):        
        reconstruction_loss = nn.functional.binary_cross_entropy(reconstructed_x, x, reduction='sum')     
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())        
        return reconstruction_loss + kl_divergence


class VAETrainer:
    """
    Write code for training Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as VAE_epoch_{}.png
    """
    def __init__(self,dataloader,encoder,decoder,loss_fn,optimizer,gpu='F'):
        self.dataloader=dataloader
        self.encoder=Encoder(Type='VAE')
        self.decoder=Decoder(Type='VAE')
        
        O = torch.optim.Adam(ParameterSelector(self.encoder, self.decoder), lr=LEARNING_RATE)
        self.optimizer=O
        self.loss_fn=loss_fn
        self.gpu=gpu
        # self.train()
        if (loss_fn!=None and dataloader!=None): self.train()
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def train(self):
        train_dataloader=self.dataloader
        # Training loop
        device = torch.device("cuda:0") if self.gpu == "T" else torch.device("cpu") 
        for epoch in range(EPOCH):
            
            train_ssim_data=[]
            train_ssim_values=[]
            train_loss=0
            self.encoder.train()
            self.decoder.train()
            for idx,data in enumerate(train_dataloader):      
                minibatch=idx+1
                noisy_inputs,clean_inputs,labels = data
                noisy_inputs.to(device)
                clean_inputs.to(device)
                                
                encoded= self.encoder(noisy_inputs)                  
                mu, logvar = encoded[:, :self.encoder.latent_dim], encoded[:, self.encoder.latent_dim:]     # latent parameters                         
                z = self.reparameterize(mu, logvar)        # Reparameterization trick                
                reconstructed_x = self.decoder(z)                                    
                loss = self.loss_fn.vae_loss(reconstructed_x, clean_inputs, mu, logvar)  # recontruction loss + KL divergence                         
                
                train_loss+=loss.item()                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                batch_ssim = calculate_ssim_batch(reconstructed_x,clean_inputs)
                train_ssim_values.extend(batch_ssim)
                train_ssim_data.extend(batch_ssim)
                
                # print ssim after every 10th minibatch
                if (minibatch) % 10 == 0:
                    with torch.no_grad():
                        avg_ssim = np.mean(train_ssim_values)
                        similarity=avg_ssim                
                        print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch+1,minibatch,train_loss/minibatch,similarity))
                        train_ssim_values = []           
            train_similarity=np.mean(train_ssim_data)
            
                
            print("----- Epoch:{}, Loss:{}, Similarity:{}".format(epoch+1,train_loss/len(train_dataloader),train_similarity))
        
                    
            # Compute embeddings after every 10 epochs and plot t-SNE
            if (epoch + 1) % 10 == 0:
                plot_tsne(train_dataloader,device,self.encoder,self.decoder,epoch,type='VAE')  
        
        # save checkpoint and print per epoch info
        save_checkpoints(epoch,loss,self.optimizer,self.encoder,self.decoder,checkpoint_dir_vae1,checkpoint_dir_vae2) 

class VAE_TRAINED:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image.
    """
    def __init__(self,gpu):
        self.gpu=gpu

    def from_path(self,sample, original, type):
        "Compute similarity score of both 'sample' and 'original' and return in float"
        checkpoint1=torch.load(checkpoint_dir_vae1)
        checkpoint2=torch.load(checkpoint_dir_vae2)
        encoder=Encoder(Type='VAE')
        decoder=Decoder(Type='VAE')
        encoder.load_state_dict(checkpoint1['model_state_dict'])
        decoder.load_state_dict(checkpoint2['model_state_dict'])
        encoder.eval()
        decoder.eval()
        # gpu='T'
        device= torch.device("cuda:0") if self.gpu == "T" else torch.device("cpu")
        
          
        inputs=sample,original
        noisy_img=data_transforms(TF.to_pil_image(read_image(sample)))
        clean_img=data_transforms(TF.to_pil_image(read_image(original)))
        
        noisy_img=noisy_img.view(1,noisy_img.size(0),noisy_img.size(1),noisy_img.size(2))
        clean_img=clean_img.view(1,clean_img.size(0),clean_img.size(1),clean_img.size(2))
        
        
        noisy_img.to(device)
        clean_img.to(device)
        encoded= encoder(noisy_img)                   
        mu, logvar = encoded[:, :encoder.latent_dim], encoded[:, encoder.latent_dim:]     # latent parameters                         
        z = VAETrainer(None, None,None,None,None).reparameterize(mu, logvar)        # Reparameterization trick                
        reconstructed_x = decoder(z)                  
                          
        if type=='SSIM': score = calculate_ssim_single(reconstructed_x,clean_img)
        else: score=peak_signal_to_noise_ratio(reconstructed_x,clean_img)
        return score










class CVAELossFn():
    """
    Write code for loss function for training Conditional Variational AutoEncoder
    """
    pass

class CVAE_Trainer:
    """
    Write code for training Conditional Variational AutoEncoder here.
    
    for each 10th minibatch use only this print statement
    print(">>>>> Epoch:{}, Minibatch:{}, Loss:{}, Similarity:{}".format(epoch,minibatch,loss,similarity))
    
    for each epoch use only this print statement
    print("----- Epoch:{}, Loss:{}, Similarity:{}")
    
    After every 5 epochs make 3D TSNE plot of logits of whole data and save the image as CVAE_epoch_{}.png
    """
    pass

class CVAE_Generator:
    """
    Write code for loading trained Encoder-Decoder from saved checkpoints for Conditional Variational Autoencoder paradigm here.
    use forward pass of both encoder-decoder to get output image conditioned to the class.
    """
    
    def save_image(digit, save_path):
        pass

    
    
    
    
    