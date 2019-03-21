from torch.backends import cudnn
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils import data
import numpy as np
import os
import argparse
import time
import datetime
import tensorflow as tf
import random
from PIL import Image


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)
    
class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
#         layers.append(nn.Conv2d(1+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
#         layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)  

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
#         layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))

def classification_loss(logit, target):
    """Compute binary or softmax cross entropy loss."""
    return F.cross_entropy(logit, target)

def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                                inputs=x,
                                grad_outputs=weight,
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def create_labels(c_org, device, c_dim=5):
    """Generate target domain labels for debugging and testing."""
    c_trg_list = []
    for i in range(c_dim):
        c_trg = label2onehot(torch.ones(c_org.size(0))*i, c_dim)
        c_trg_list.append(c_trg.to(device))
    return c_trg_list

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def update_lr(g_lr, d_lr, g_optimizer, d_optimizer):
    """Decay learning rates of the generator and discriminator."""
    for param_group in g_optimizer.param_groups:
        param_group['lr'] = g_lr
    for param_group in d_optimizer.param_groups:
        param_group['lr'] = d_lr

def restore_model(resume_iters, G, D, model_save_dir):
    """Restore the trained generator and discriminator."""
    print('Loading the trained models from step {}...'.format(resume_iters))
    G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(resume_iters))
    D_path = os.path.join(model_save_dir, '{}-D.ckpt'.format(resume_iters))
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    return G, D


def get_loader(image_dir, crop_size=178, image_size=128, 
               batch_size=16, mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
    
def train():
    """Train StarGAN within a single dataset."""
    # For fast training.
    cudnn.benchmark = True

    # Set parameter.
    image_dir = "data\\RaFD\\train"
    model_save_dir = "test\\models"
    log_dir = "test\\logs"
    sample_dir =  "test\\samples"

    crop_size=256
    image_size=64
    batch_size=16
    c_dim=3
    num_iters = 200000
    g_lr = 0.0001
    d_lr = 0.0001
    log_step = 10
    sample_step = 1000
    model_save_step = 1000
    resume_iters = None
    use_tensorboard = True

    # ほぼ変更しないパラメータ
    mode='train'
    num_workers=1
    lambda_cls = 1
    lambda_rec = 10
    lambda_gp = 10
    n_critic = 5
    beta1 = 0.5
    beta2 = 0.999
    lr_update_step = 1000
    num_iters_decay = 100000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create directories if not exist.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    logger = Logger(log_dir)

    G = Generator(conv_dim=64, c_dim=c_dim, repeat_num=6)
    D = Discriminator(image_size=image_size, conv_dim=64, c_dim=c_dim, repeat_num=6) 

    g_optimizer = torch.optim.Adam(G.parameters(), 0.0001, [0.5, 0.999])
    d_optimizer = torch.optim.Adam(D.parameters(), 0.0001, [0.5, 0.999])
    G.to(device)
    D.to(device)

    # data loaderの設定
    data_loader = get_loader(image_dir, crop_size, image_size, batch_size, mode, num_workers)

    # 学習途中生成画像評価用のデータ
    data_iter = iter(data_loader)
    x_fixed, c_org = next(data_iter)
    x_fixed = x_fixed.to(device)
    c_fixed_list = create_labels(c_org, device, c_dim)

    # Learning rate cache for decaying.
    g_lr_dcay = g_lr
    d_lr_dcay = d_lr

    # Start training from scratch or resume training.
    start_iters = 0
    if resume_iters:
        start_iters = resume_iters
        restore_model(resume_iters, G, D, model_save_dir)

    # Start training.
    print('Start training...')
    start_time = time.time()
    for i in range(start_iters, num_iters):

        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #

        # Fetch real images and labels.
        try:
            x_real, label_org = next(data_iter)
        except:
            data_iter = iter(data_loader)
            x_real, label_org = next(data_iter)

        # Generate target domain labels randomly.
        rand_idx = torch.randperm(label_org.size(0))
        label_trg = label_org[rand_idx]

        c_org = label2onehot(label_org, c_dim)
        c_trg = label2onehot(label_trg, c_dim)

        x_real = x_real.to(device)           # Input images.
        c_org = c_org.to(device)             # Original domain labels.
        c_trg = c_trg.to(device)             # Target domain labels.
        label_org = label_org.to(device)     # Labels for computing classification loss.
        label_trg = label_trg.to(device)     # Labels for computing classification loss.

        # =================================================================================== #
        #                             2. Train the discriminator                              #
        # =================================================================================== #

        # Compute loss with real images.
        out_src, out_cls = D(x_real)
        d_loss_real = - torch.mean(out_src)
        d_loss_cls = classification_loss(out_cls, label_org)

        # Compute loss with fake images.
        x_fake = G(x_real, c_trg)
        out_src, out_cls = D(x_fake.detach())
        d_loss_fake = torch.mean(out_src)

        # Compute loss for gradient penalty.
        alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)
        x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        out_src, _ = D(x_hat)
        d_loss_gp = gradient_penalty(out_src, x_hat, device)

        # Backward and optimize.
        d_loss = d_loss_real + d_loss_fake + lambda_cls * d_loss_cls + lambda_gp * d_loss_gp
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Logging.
        loss = {}
        loss['D/loss_real'] = d_loss_real.item()
        loss['D/loss_fake'] = d_loss_fake.item()
        loss['D/loss_cls'] = d_loss_cls.item()
        loss['D/loss_gp'] = d_loss_gp.item()
        
        # =================================================================================== #
        #                               3. Train the generator                                #
        # =================================================================================== #
        if (i+1) % n_critic == 0:
            # Original-to-target domain.
            x_fake = G(x_real, c_trg)
            out_src, out_cls = D(x_fake)
            g_loss_fake = - torch.mean(out_src)
            g_loss_cls = classification_loss(out_cls, label_trg)

            # Target-to-original domain.
            x_reconst = G(x_fake, c_org)
            g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

            # Backward and optimize.
            g_loss = g_loss_fake + lambda_rec * g_loss_rec + lambda_cls * g_loss_cls
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Logging.
            loss['G/loss_fake'] = g_loss_fake.item()
            loss['G/loss_rec'] = g_loss_rec.item()
            loss['G/loss_cls'] = g_loss_cls.item()

        # =================================================================================== #
        #                                 4. Miscellaneous                                    #
        # =================================================================================== #
        # Print out training information.
        if (i+1) % log_step == 0:
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, num_iters)
            for tag, value in loss.items():
                log += ", {}: {:.4f}".format(tag, value)
            print(log)

            if use_tensorboard:
                for tag, value in loss.items():
                    logger.scalar_summary(tag, value, i+1)

        # Translate fixed images for debugging.
        if (i+1) % sample_step == 0:
            with torch.no_grad():
                x_fake_list = [x_fixed]
                for c_fixed in c_fixed_list:
                    x_fake_list.append(G(x_fixed, c_fixed))
                x_concat = torch.cat(x_fake_list, dim=3)
                sample_path = os.path.join(sample_dir, '{}-images.jpg'.format(i+1))
                save_image(denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(sample_path))

        # Save model checkpoints.
        if (i+1) % model_save_step == 0:
            G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(i+1))
            D_path = os.path.join(model_save_dir, '{}-D.ckpt'.format(i+1))
            torch.save(G.state_dict(), G_path)
            torch.save(D.state_dict(), D_path)
            print('Saved model checkpoints into {}...'.format(model_save_dir))

        # Decay learning rates.
        if (i+1) % lr_update_step == 0 and (i+1) > (num_iters - num_iters_decay):
            g_lr_dcay -= (g_lr / float(num_iters_decay))
            d_lr_dcay -= (d_lr / float(num_iters_decay))
            update_lr(g_lr_dcay, d_lr_dcay)
            print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr_dcay, d_lr_dcay))

                
def test_st(c_trg):
    """Translate images using StarGAN trained on a single dataset."""
     # For fast training.
    cudnn.benchmark = True

    # Set parameter.
    image_dir = "test\\inp"
    model_save_dir = "C:/Users/Ichiyama/Anaconda3/envs/tensorflow/work/facialexp_conv/models/emo2Img"
    result_dir = "test\\results"

    crop_size=256
    image_size=64
    batch_size = 1
    test_iters = 200000
    num_workers = 1
    c_dim = 3 # 学習時の感情クラス数：ターゲットクラスをone-hotベクトル表現にするために必要

    # ほぼ変更しないパラメータ
    mode='test_st'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create directories if not exist.
    if not os.path.exists(model_save_dir):
        print('model dir が存在しません')
        return
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Generator作成
    G = Generator(conv_dim=64, c_dim=c_dim, repeat_num=6)
    G.to(device)

    # Load the trained generator.
    G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(test_iters))
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    # 前処理をdata_loaderにやらせるため設定：そのため例のフォルダ構成をとる必要がある
    data_loader = get_loader(image_dir, crop_size, image_size, batch_size, mode, num_workers)
    
    with torch.no_grad():
        for i, (x_real, c_dummy) in enumerate(data_loader):

            # Prepare input images and target domain labels.
            c_trg = label2onehot(c_trg, c_dim) # class no. ⇒ one hot vector
            x_real = x_real.to(device)
            c_trg = c_trg.to(device)
    
            # Translate images.
            x_fake = G(x_real, c_trg)

            # Save the translated images.
            result_path = os.path.join(result_dir, 'result.bmp')
            save_image(denorm(x_fake.data.cpu()), result_path, nrow=1, padding=0)
            print('Saved real and fake images into {}...'.format(result_path))           
                            
def test_mv(G, image_dir, oupImage, c_trg, c_dim):
    from PIL import Image

    """Translate images using StarGAN trained on a single dataset."""
     # For fast training.
    cudnn.benchmark = True

    crop_size=256
    image_size=256
    batch_size = 1
    num_workers = 1

    # 元々の画像のクラス
    c_org = torch.Tensor([1])

    # ほぼ変更しないパラメータ
    mode='test_mv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 前処理をdata_loaderにやらせるため設定：そのため例のフォルダ構成をとる必要がある
    data_loader = get_loader(image_dir, crop_size, image_size, batch_size, mode, num_workers)
    
    with torch.no_grad():
        for i, (x_real, c_dummy) in enumerate(data_loader):
            
            # Prepare input images and target domain labels.
            c_trg = label2onehot(c_trg, c_dim) # class no. ⇒ one hot vector
            c_org = label2onehot(c_org, c_dim) # class no. ⇒ one hot vector
            x_real = x_real.to(device)
            c_trg = c_trg.to(device)
            c_org = c_org.to(device)

            images = []
    
            for j in range(11):
            
                # Translate images.
                intensity = 0.1*j
                x_fake = G(x_real, c_trg*intensity + c_org*(1-intensity))

                im = denorm(x_fake.data.cpu()).numpy()
                im = np.squeeze(im)*255
                im = im.transpose(1,2,0)
                im = Image.fromarray(im.astype(np.uint8))
                images.append(im)

            images[0].save(oupImage, save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)
            # print('Saved real and fake images into {}...'.format(oupImage))           

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test_st', 'test_mv'])

    config = parser.parse_args()
    print(config)
   
    if config.mode == 'train':
        train()
    elif config.mode == 'test_st':
        test_st(torch.Tensor([2]))
    elif config.mode == 'test_mv':
        test_mv(torch.Tensor([1]), torch.Tensor([2]))
        