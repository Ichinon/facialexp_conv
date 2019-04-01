import numpy as np
import torch
from pixyz.distributions import Deterministic, DataDistribution
from pixyz.losses import AdversarialJensenShannon
from pixyz.models import Model
from pixyz.utils import get_dict_values, detach_dict
from torchvision import transforms as T
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils import data
import os
import time
import datetime
from torch.backends import cudnn
import tensorflow as tf

import torch.nn as nn

src_dir = os.path.dirname(__file__)

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
    
class Generator(Deterministic):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__(cond_var=["z", "c"], var=["x"])

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, z, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, z.size(2), z.size(3))
        z = torch.cat([z, c], dim=1)
        return {"x": self.main(z)}

class Discriminator(Deterministic):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__(cond_var=["x"], var=["out_src", "out_cls"])
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
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
        return {"out_src":out_src.view(out_src.size(0), out_src.size(1)), "out_cls":out_cls.view(out_cls.size(0), out_cls.size(1))}

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

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

class StarGanLoss(AdversarialJensenShannon):

    def __init__(self, p, q, discriminator, input_var=None):
        super().__init__(p, q, discriminator, input_var=input_var)

        self.bce_loss = nn.BCELoss()
        self.lambda_cls = 1
        self.label_org = {}
        self.label_trg = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def set_label_org(self, label_org):
        self.label_org = label_org
        return
    
    def set_label_trg(self, label_trg):
        self.label_trg = label_trg
        return

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def estimate(self, x={}, return_dict=False, **kwargs):
        if not(set(list(x.keys())) >= set(self._input_var)):
            raise ValueError("Input keys are not valid, got {}.".format(list(x.keys())))

        loss, x = self._get_estimated_value(x, **kwargs)

        if return_dict:
            return loss, x

        return loss
    def _get_estimated_value(self, x, discriminator=False, **kwargs):
        batch_size = get_dict_values(x, self._p1.input_var[0])[0].shape[0]

        # x1_dict: x_real "x"
        # x2_dict: x_fake "x"

        # sample x from p1:
        x_dict = get_dict_values(x, self._p1.input_var, True)
        if self._p1_data_dist:
            x1_dict = x_dict
        else:
            x1_dict = self._p1.sample(x_dict, batch_size=batch_size)
            x1_dict = get_dict_values(x1_dict, self.d.input_var, True)

        # sample x from p2
        x_dict = get_dict_values(x, self._p2.input_var, True)
        # x_fake = p(x|z)p(z)
        x2_dict = self._p2.sample(x_dict, batch_size=batch_size)
        x2_dict = get_dict_values(x2_dict, self.d.input_var, True)

        if discriminator:
            # sample y from x1
            # D(x_real)
            y1_dict = self.d.sample(x1_dict)
            y1 = get_dict_values(y1_dict, self.d.var)

            # sample y from x2
            # D(x_fake)
            y2_dict = self.d.sample(detach_dict(x2_dict))
            y2 = get_dict_values(y2_dict, self.d.var)

            d_loss_real, d_loss_fake, d_loss_cls = self.d_loss(y1, y2, batch_size)

            alpha = torch.rand(x1_dict["x"].size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x1_dict["x"].data + (1 - alpha) * x2_dict["x"].data).requires_grad_(True)
            y3_dict = self.d.sample({"x":x_hat})
            d_loss_gp = self.gradient_penalty(y3_dict["out_src"], x_hat, self.device)

            return {"d_loss_real":d_loss_real, "d_loss_fake":d_loss_fake, "d_loss_cls":d_loss_cls, "d_loss_gp":d_loss_gp}, x

        # sample y from x1
        y1_dict = self.d.sample(x1_dict)
        # sample y from x2
        y2_dict = self.d.sample(x2_dict)

        y1 = get_dict_values(y1_dict, self.d.var)
        y2 = get_dict_values(y2_dict, self.d.var)

        g_loss_fake, g_loss_cls = self.g_loss(y1, y2, batch_size)

        c_dim = x_dict["c"].size()[1]
        c_org = self.label2onehot(self.label_org, c_dim).to(self.device)
        x3_dict = {"z":x2_dict["x"], "c":c_org}
        x_reconst = self._p2.sample(x3_dict)
        g_loss_rec = torch.mean(torch.abs(x1_dict["x"] - x_reconst["x"]))

        return {"g_loss_fake":g_loss_fake, "g_loss_cls":g_loss_cls, "g_loss_rec":g_loss_rec}, x

    def d_loss(self, y1, y2, batch_size):

        # y1[0]:out_src(real)
        # y1[1]:out_cls(real)
        # y2[0]:out_src(fake)
        # y2[1]:out_cls(fake)

        d_loss_real = - torch.mean(y1[0])
        d_loss_fake = torch.mean(y2[0])
        # d_loss_fake = d_loss_fake.detach() # Gに伝搬しないように

        d_loss_cls = F.cross_entropy(y1[1], self.label_org)

        return d_loss_real, d_loss_fake, d_loss_cls

    def g_loss(self, y1, y2, batch_size):

        # y1[0]:out_src(real)
        # y1[1]:out_cls(real)
        # y2[0]:out_src(fake)
        # y2[1]:out_cls(fake)

        g_loss_fake = -torch.mean(y2[0])
        g_loss_cls = F.cross_entropy(y2[1], self.label_trg)

        return g_loss_fake, g_loss_cls
    
    def gradient_penalty(self, y, x, device):
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

class StarGAN(Model):
    """
    Generative Adversarial Network
    """
    def __init__(self, p_data, generator, c, image_dir, mode, discriminator={}, image_size=256, batch_size=16, c_dim=8, resume_iters=None, model_save_dir=None):
        # 学習データセット前処理パラメータ：画像のクロップサイズ
        self.crop_size = 256
        # 学習データセット前処理パラメータ：クロップ後に↓サイズにリサイズ
        self.image_size = image_size
        # ミニバッチサイズ
        self.batch_size = batch_size
        # 感情クラス数
        self.c_dim = c_dim
        # 学習回数
        self.num_iters = 200000
        # 学習開始iteration回数
        self.resume_iters = resume_iters
        # loss表示頻度：↓回に1回表示
        self.log_step = 100
        # 学習途中の画像生成実施頻度：↓回に1回実施
        self.sample_step = 1000
        # モデル保存頻度：↓回に１回実施
        self.model_save_step = 1000
        # 学習データセットパス
        self.image_dir = image_dir
        # 出力先親フォルダパス：この下にモデル・ログ・サンプルをフォルダを作成し出力
        self.out_parent_dir = os.path.join(os.path.dirname(__file__), "../../")
        # 学習時かどうかのフラグ：データ拡張実施有無に利用
        self.mode = mode
        # 実行環境指示
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 学習パラメータ
        self.lambda_cls = 1
        self.lambda_rec = 10
        self.lambda_gp = 10
        self.lr_update_step = 1000
        self.num_iters_decay = 100000
        self.g_lr =  0.0001
        self.d_lr =  0.0001
        self.n_critic = 5
        self.beta1 = 0.5
        self.beta2 = 0.999

        self.G = generator
        self.p = (self.G*c).to(self.device)
        self.g_optimizer = torch.optim.Adam(self.p.parameters(), self.g_lr, [self.beta1, self.beta2])

        if self.mode == 'train':
            self.D = discriminator
            self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        
        # モデル保存先パス
        if model_save_dir:
            self.model_save_dir = model_save_dir
        else:
            self.model_save_dir = os.path.join(self.out_parent_dir, "models")
        # 学習途中の生成結果保存先パス
        self.sample_dir = os.path.join(self.out_parent_dir, "samples")
        # ログ保存先パス
        self.log_dir = os.path.join(self.out_parent_dir, "logs")
        # tensorboard保存用ロガー
        self.logger = Logger(self.log_dir)
        self.use_tensorboard = True
        
        # Load model
        self.resume_iters = resume_iters
        if self.resume_iters:
            self.G = self.restore_Gmodel(self.G)
            if self.mode == 'train':
                self.D = self.restore_Dmodel(self.D)
            self.start_iters = self.resume_iters
        else:
            self.start_iters = 0

        # StarGANのLoss： p_data:学習データセット, p:generator, discriminator:discriminator
        if self.mode == 'train':
            self.loss_AJS = StarGanLoss(p_data, self.p, self.D)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr
    
    def restore_Gmodel(self, G):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(self.resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(self.resume_iters))
        G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        return G
    
    def restore_Dmodel(self, D):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(self.resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(self.resume_iters))
        D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        return D

    def train(self):

        # For fast training.
        cudnn.benchmark = True

        # Create directories if not exist.
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        data_loader = get_loader(self.image_dir, self.crop_size, self.image_size, self.batch_size, self.mode, num_workers=1)

        # 学習途中生成画像評価用のデータ
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = create_labels(c_org, self.device, self.c_dim)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        print('Start training...')
        start_time = time.time()
        for i in range(self.start_iters, self.num_iters):
### 1. Preprocess ###
            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)
            
            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = label2onehot(label_org, self.c_dim)
            c_trg = label2onehot(label_trg, self.c_dim)

            x_real = x_real.to(self.device)           # Input images.
            c_org = c_org.to(self.device)             # Original domain labels.
            c_trg = c_trg.to(self.device)             # Target domain labels.
            label_org = label_org.to(self.device)     # Labels for computing classification loss.
            label_trg = label_trg.to(self.device)     # Labels for computing classification loss.

### 2. Train the discriminator ###
            # Loss計算
            self.loss_AJS.set_label_org(label_org)
            self.loss_AJS.set_label_trg(label_trg)
            dloss_AJS = self.loss_AJS.estimate({"x": x_real, "z": x_real, "c":c_trg}, discriminator=True)
            d_loss_real = dloss_AJS["d_loss_real"]
            d_loss_fake = dloss_AJS["d_loss_fake"]
            d_loss_cls = dloss_AJS["d_loss_cls"]
            d_loss_gp = dloss_AJS["d_loss_gp"]
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            # 誤差伝搬
            self.g_optimizer.zero_grad()
            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()

### 3. Train the generator ###
            if (i+1) % self.n_critic == 0:
                gloss_AJS = self.loss_AJS.estimate({"x": x_real, "z": x_real, "c":c_trg}, discriminator=False)
                g_loss_fake = gloss_AJS["g_loss_fake"]
                g_loss_cls = gloss_AJS["g_loss_cls"]
                g_loss_rec = gloss_AJS["g_loss_rec"]
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                g_loss = g_loss_fake + self.lambda_cls * g_loss_cls +  self.lambda_rec * g_loss_rec
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

### 4. Miscellaneous ###
            # ログ出力
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)
            
            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(G(x_fixed, c_fixed)["x"])
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > self.num_iters_decay:
                g_lr -= (self.g_lr / float(self.num_iters - self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters - self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

        print("End training...")

        return

    def test(self, oupImage, c_trg, c_org, c_dim):
        from PIL import Image

        """Translate images using StarGAN trained on a single dataset."""
        # For fast training.
        cudnn.benchmark = True

        num_workers = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = 1

        # 前処理をdata_loaderにやらせるため設定：そのため例のフォルダ構成をとる必要がある
        data_loader = get_loader(self.image_dir, self.crop_size, self.image_size, batch_size, self.mode, num_workers)
        
        with torch.no_grad():
            for i, (x_real, _) in enumerate(data_loader):
                
                # Prepare input images and target domain labels.
                c_trg = label2onehot(c_trg, c_dim) # class no. ⇒ one hot vector
                c_org = label2onehot(c_org, c_dim) # class no. ⇒ one hot vector
                x_real = x_real.to(device)
                c_trg = c_trg.to(device)
                c_org = c_org.to(device)

                images = []
                
                if self.mode == 'test_mv':
                    im_num = 11
                else:
                    im_num = 1

                for j in range(im_num):
                
                    # Translate images.
                    intensity = 0.1*j
                    x_fake = self.G(x_real,c_trg*(1-intensity) + c_org*intensity)

                    im = denorm(x_fake["x"].data.cpu()).numpy()
                    im = np.squeeze(im)*255
                    im = im.transpose(1,2,0)
                    im = Image.fromarray(im.astype(np.uint8))
                    images.insert(0,im)

                images[0].save(oupImage, save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)
                # print('Saved real and fake images into {}...'.format(oupImage))           

    def test_images(self, result_dir, c_dim, batch_size):
        from torchvision.utils import save_image

        """Translate images using StarGAN trained on a single dataset."""
        # For fast training.
        cudnn.benchmark = True

        num_workers = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 前処理をdata_loaderにやらせるため設定：そのため例のフォルダ構成をとる必要がある
        data_loader = get_loader(self.image_dir, self.crop_size, self.image_size, batch_size, self.mode, num_workers)
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):
                
                # Prepare input images and target domain labels.
                x_real = x_real.to(device)
                c_trg_list = create_labels(c_org, device, self.c_dim)

                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake = self.G(x_real, c_trg)
                    x_fake_list.append(x_fake["x"])

                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(result_dir, '{}-images.jpg'.format(i+1))
                save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

if __name__ == '__main__':

###############################
##### Training #####
###############################
    
### parameter
    # 学習用データパス
    image_dir = os.path.join(os.path.dirname(__file__), "../../data/team1_dataset_500")
    # 感情クラス数
    c_dim = 7
    # 学習画像サイズ
    image_size = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode = 'train'

### distribution

    # generator
    G = Generator(conv_dim=64, c_dim=c_dim, repeat_num=6).to(device)
    # discriminator
    D = Discriminator(image_size=image_size, conv_dim=64, c_dim=c_dim, repeat_num=6).to(device)
    # real data
    p_data = DataDistribution(["x"]).to(device)
    # target emotion class
    c = DataDistribution(["c"]).to(device)

### model
    model = StarGAN(p_data, G, c, image_dir, mode, discriminator=D, c_dim=c_dim, image_size=image_size)
    
### training
    model.train()

