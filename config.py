"""Global configuration."""
#----------------------------------------------------------------------------
# Directories.
celeba_image_dir='data/celeba/images'   # 'data/celeba/images'
attr_path = 'data/celeba/list_attr_celeba.txt'  # 'data/celeba/list_attr_celeba.txt'
rafd_image_dir='inp/test'   # 'data/RaFD/train'
log_dir='stargan/logs'  # 'stargan/logs'
model_save_dir='models/emo2Img'     # 'stargan/models'
sample_dir='stargan/samples'        # 'stargan/samples'
result_dir='res'    # 'stargan/results'
#----------------------------------------------------------------------------
# Miscellaneous.
num_workers=1
mode='test'    # choices=['train', 'test'])
use_tensorboard=False   # True    
#----------------------------------------------------------------------------
# Training configuration.
dataset='RaFD'    # choices=['CelebA', 'RaFD', 'Both'])
selected_attrs=['neg', 'neu', 'pos']    # 'selected attributes for the CelebA dataset' ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
batch_size=16    # 'mini-batch size')
num_iters=200000    # 'number of total iterations for training D')
num_iters_decay=100000    # 'number of iterations for decaying lr')
g_lr=0.0001    # 'learning rate for G')
d_lr=0.0001    # 'learning rate for D')
n_critic=5    # 'number of D updates per each G update')
beta1=0.5    # 'beta1 for Adam optimizer')
beta2=0.999    # 'beta2 for Adam optimizer')
resume_iters=None    # 'resume training from this step')
#----------------------------------------------------------------------------
# Model configuration.
c_dim=3     # 5 'dimension of domain labels (1st dataset)')
c2_dim=8    # 8 'dimension of domain labels (2nd dataset)')
celeba_crop_size=178    # 'crop size for the CelebA dataset')
rafd_crop_size=256    # 'crop size for the RaFD dataset')
image_size=64   # 128 'image resolution'
g_conv_dim=64    # 'number of conv filters in the first layer of G')
d_conv_dim=64    # 'number of conv filters in the first layer of D')
g_repeat_num=6    # 'number of residual blocks in G')
d_repeat_num=6    # 'number of strided conv layers in D')
lambda_cls=1    # 'weight for domain classification loss')
lambda_rec=10    # 'weight for reconstruction loss')
lambda_gp=10    # 'weight for gradient penalty')
#----------------------------------------------------------------------------
# Test configuration.
test_iters=200000    # 'test model from this step')
#----------------------------------------------------------------------------
# Step size.
log_step=10
sample_step=1000
model_save_step=10000
lr_update_step=1000
#----------------------------------------------------------------------------
