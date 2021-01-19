import os

root_dir = '/nas/home/nbonettini/projects/icpr-benford-gan/'

tmp_path = '/home/nbonettini/.temp'

dogan_generated_root = '/nas/public/dataset/DoGANs/Generated'
dogan_pristine_root = '/nas/public/dataset/DoGANs/Pristine'
stylegan2_generated_root = '/nas/home/nbonettini/projects/jstsp-benford-gan/additional_gan_images/stylegan2'
ffhq_pristine_root = '/nas/home/nbonettini/projects/jstsp-benford-gan/additional_gan_images/ffhq'
lsun_prisine_root = '/nas/home/nbonettini/projects/jstsp-benford-gan/additional_gan_images/lsun'

dataset_root = {
    'apple2orange_gan': os.path.join(dogan_generated_root, 'CycleGAN', 'apple2orange'),
    'orange2apple_gan': os.path.join(dogan_generated_root, 'CycleGAN', 'orange2apple'),
    'photo2ukiyoe_gan': os.path.join(dogan_generated_root, 'CycleGAN', 'photo2ukiyoe'),
    'summer2winter_gan': os.path.join(dogan_generated_root, 'CycleGAN', 'summer2winter'),
    'horse2zebra_gan': os.path.join(dogan_generated_root, 'CycleGAN', 'horse2zebra'),
    'photo2cezanne_gan': os.path.join(dogan_generated_root, 'CycleGAN', 'photo2cezanne'),
    'photo2vangogh_gan': os.path.join(dogan_generated_root, 'CycleGAN', 'photo2vangogh'),
    'winter2summer_gan': os.path.join(dogan_generated_root, 'CycleGAN', 'winter2summer'),
    'monet2photo_gan': os.path.join(dogan_generated_root, 'CycleGAN', 'monet2photo'),
    'photo2monet_gan': os.path.join(dogan_generated_root, 'CycleGAN', 'photo2monet'),
    'zebra2horse_gan': os.path.join(dogan_generated_root, 'CycleGAN', 'zebra2horse'),
    'facades_gan': os.path.join(dogan_generated_root, 'CycleGAN', 'facades'),
    'cityscapes_gan': os.path.join(dogan_generated_root, 'CycleGAN', 'cityscapes'),
    'sats_gan': os.path.join(dogan_generated_root, 'CycleGAN', 'sats'),
    'celeb256_gan': os.path.join(dogan_generated_root, 'GGAN256', 'celeba256'),
    'lsun_bedroom_gan': os.path.join(dogan_generated_root, 'GGAN256', 'lsun_bedroom'),
    'lsun_bridge_gan': os.path.join(dogan_generated_root, 'GGAN256', 'lsun_bridge'),
    'lsun_churchoutdoor_gan': os.path.join(dogan_generated_root, 'GGAN256', 'lsun_churchoutdoor'),
    'lsun_kitchen_gan': os.path.join(dogan_generated_root, 'GGAN256', 'lsun_kitchen'),
    'lsun_tower_gan': os.path.join(dogan_generated_root, 'GGAN256', 'lsun_tower'),

    'apple2orange_orig': os.path.join(dogan_pristine_root, 'CycleGAN', 'apple2orange'),
    'orange2apple_orig': os.path.join(dogan_pristine_root, 'CycleGAN', 'orange2apple'),
    'photo2ukiyoe_orig': os.path.join(dogan_pristine_root, 'CycleGAN', 'photo2ukiyoe'),
    'summer2winter_orig': os.path.join(dogan_pristine_root, 'CycleGAN', 'summer2winter'),
    'horse2zebra_orig': os.path.join(dogan_pristine_root, 'CycleGAN', 'horse2zebra'),
    'photo2cezanne_orig': os.path.join(dogan_pristine_root, 'CycleGAN', 'photo2cezanne'),
    'photo2vangogh_orig': os.path.join(dogan_pristine_root, 'CycleGAN', 'photo2vangogh'),
    'winter2summer_orig': os.path.join(dogan_pristine_root, 'CycleGAN', 'winter2summer'),
    'monet2photo_orig': os.path.join(dogan_pristine_root, 'CycleGAN', 'monet2photo'),
    'photo2monet_orig': os.path.join(dogan_pristine_root, 'CycleGAN', 'photo2monet'),
    'zebra2horse_orig': os.path.join(dogan_pristine_root, 'CycleGAN', 'zebra2horse'),
    'facades_orig': os.path.join(dogan_pristine_root, 'CycleGAN', 'facades'),
    'cityscapes_orig': os.path.join(dogan_pristine_root, 'CycleGAN', 'cityscapes'),
    'sats_orig': os.path.join(dogan_pristine_root, 'CycleGAN', 'sats'),
    'celeb256_orig': os.path.join(dogan_pristine_root, 'GGAN256', 'celeba256'),
    'lsun_bedroom_orig': os.path.join(dogan_pristine_root, 'GGAN256', 'lsun_bedroom'),
    'lsun_bridge_orig': os.path.join(dogan_pristine_root, 'GGAN256', 'lsun_bridge'),
    'lsun_churchoutdoor_orig': os.path.join(dogan_pristine_root, 'GGAN256', 'lsun_churchoutdoor'),
    'lsun_kitchen_orig': os.path.join(dogan_pristine_root, 'GGAN256', 'lsun_kitchen'),
    'lsun_tower_orig': os.path.join(dogan_pristine_root, 'GGAN256', 'lsun_tower'),
}

dataset_root_faces = {'celeb256_gan': os.path.join(dogan_generated_root, 'GGAN256', 'celeba256'),
                      'glow_black_hair_gan': os.path.join(dogan_generated_root, 'glow', 'Black_Hair'),
                      'glow_blond_hair_gan': os.path.join(dogan_generated_root, 'glow', 'Blond_Hair'),
                      'glow_brown_hair_gan': os.path.join(dogan_generated_root, 'glow', 'Brown_Hair'),
                      'glow_male_gan': os.path.join(dogan_generated_root, 'glow', 'Male'),
                      'glow_smiling_gan': os.path.join(dogan_generated_root, 'glow', 'Smiling'),
                      'stargan_black_hair_gan': os.path.join(dogan_generated_root, 'starGAN', 'Black_Hair'),
                      'stargan_blond_hair_gan': os.path.join(dogan_generated_root, 'starGAN', 'Blond_Hair'),
                      'stargan_brown_hair_gan': os.path.join(dogan_generated_root, 'starGAN', 'Brown_Hair'),
                      'stargan_male_gan': os.path.join(dogan_generated_root, 'starGAN', 'Male'),
                      'stargan_smiling_gan': os.path.join(dogan_generated_root, 'starGAN', 'Smiling'),
                      'stylegan2-0.5_gan': os.path.join(stylegan2_generated_root, 'config-f-psi-0.5'),
                      'stylegan2-1_gan': os.path.join(stylegan2_generated_root, 'config-f-psi-1'),

                      'celeb256_orig': os.path.join(dogan_pristine_root, 'GGAN256', 'celeba256'),
                      'stylegan2_orig': ffhq_pristine_root,
                      }

dataset_ext = {
    'apple2orange_gan': 'png',
    'orange2apple_gan': 'png',
    'photo2ukiyoe_gan': 'png',
    'summer2winter_gan': 'png',
    'horse2zebra_gan': 'png',
    'photo2cezanne_gan': 'png',
    'photo2vangogh_gan': 'png',
    'winter2summer_gan': 'png',
    'monet2photo_gan': 'png',
    'photo2monet_gan': 'png',
    'zebra2horse_gan': 'png',
    'facades_gan': 'png',
    'cityscapes_gan': 'png',
    'sats_gan': 'png',
    'lsun_bedroom_gan': 'png',
    'lsun_bridge_gan': 'png',
    'lsun_churchoutdoor_gan': 'png',
    'lsun_kitchen_gan': 'png',
    'lsun_tower_gan': 'png',

    'apple2orange_orig': 'png',
    'orange2apple_orig': 'png',
    'photo2ukiyoe_orig': 'png',
    'summer2winter_orig': 'png',
    'horse2zebra_orig': 'png',
    'photo2cezanne_orig': 'png',
    'photo2vangogh_orig': 'png',
    'winter2summer_orig': 'png',
    'monet2photo_orig': 'png',
    'photo2monet_orig': 'png',
    'zebra2horse_orig': 'png',
    'facades_orig': 'jpg',
    'cityscapes_orig': 'jpg',
    'sats_orig': 'jpg',
    'lsun_bedroom_orig': 'png',
    'lsun_bridge_orig': 'png',
    'lsun_churchoutdoor_orig': 'png',
    'lsun_kitchen_orig': 'png',
    'lsun_tower_orig': 'png',
}

dataset_ext_faces = {'celeb256_gan': 'png',
                     'glow_black_hair_gan': 'png',
                     'glow_blond_hair_gan': 'png',
                     'glow_brown_hair_gan': 'png',
                     'glow_male_gan': 'png',
                     'glow_smiling_gan': 'png',
                     'stargan_black_hair_gan': 'png',
                     'stargan_blond_hair_gan': 'png',
                     'stargan_brown_hair_gan': 'png',
                     'stargan_male_gan': 'png',
                     'stargan_smiling_gan': 'png',
                     'stylegan2-0.5_gan': 'png',
                     'stylegan2-1_gan': 'png',

                     'celeb256_orig': 'png',
                     'stylegan2_orig': 'png',
                     }

dataset_label_faces = {'celeb256_gan': 0,
                       'glow_black_hair_gan': 1,
                       'glow_blond_hair_gan': 2,
                       'glow_brown_hair_gan': 3,
                       'glow_male_gan': 4,
                       'glow_smiling_gan': 5,
                       'stargan_black_hair_gan': 6,
                       'stargan_blond_hair_gan': 7,
                       'stargan_brown_hair_gan': 8,
                       'stargan_male_gan': 9,
                       'stargan_smiling_gan': 10,
                       'stylegan2-0.5_gan': 11,
                       'stylegan2-1_gan': 12,
                       'celeb256_orig': 13,
                       'stylegan2_orig': 14,
                       }

dataset_label = {
    'apple2orange_gan': 0,
    'orange2apple_gan': 0,
    'photo2ukiyoe_gan': 1,
    'summer2winter_gan': 2,
    'horse2zebra_gan': 3,
    'photo2cezanne_gan': 4,
    'photo2vangogh_gan': 5,
    'winter2summer_gan': 2,
    'monet2photo_gan': 6,
    'photo2monet_gan': 6,
    'zebra2horse_gan': 3,
    'facades_gan': 7,
    'cityscapes_gan': 8,
    'sats_gan': 9,
    'lsun_bedroom_gan': 11,
    'lsun_bridge_gan': 12,
    'lsun_churchoutdoor_gan': 13,
    'lsun_kitchen_gan': 14,
    'lsun_tower_gan': 15,
    'apple2orange_orig': 0,
    'orange2apple_orig': 0,
    'photo2ukiyoe_orig': 1,
    'summer2winter_orig': 2,
    'horse2zebra_orig': 3,
    'photo2cezanne_orig': 4,
    'photo2vangogh_orig': 5,
    'winter2summer_orig': 2,
    'monet2photo_orig': 6,
    'photo2monet_orig': 6,
    'zebra2horse_orig': 3,
    'facades_orig': 7,
    'cityscapes_orig': 8,
    'sats_orig': 9,
    'lsun_bedroom_orig': 11,
    'lsun_bridge_orig': 12,
    'lsun_churchoutdoor_orig': 13,
    'lsun_kitchen_orig': 14,
    'lsun_tower_orig': 15,
}

gan_orig_map_faces = {'celeb256_gan': 'celeb256_orig',
                      'glow_black_hair_gan': 'celeb256_orig',
                      'glow_blond_hair_gan': 'celeb256_orig',
                      'glow_brown_hair_gan': 'celeb256_orig',
                      'glow_male_gan': 'celeb256_orig',
                      'glow_smiling_gan': 'celeb256_orig',
                      'stargan_black_hair_gan': 'celeb256_orig',
                      'stargan_blond_hair_gan': 'celeb256_orig',
                      'stargan_brown_hair_gan': 'celeb256_orig',
                      'stargan_male_gan': 'celeb256_orig',
                      'stargan_smiling_gan': 'celeb256_orig',
                      'stylegan2-0.5_gan': 'stylegan2_orig',
                      'stylegan2-1_gan': 'stylegan2_orig',
                      }

popt_dict = {'scale': 0, 'alpha': 1, 'beta': 2}

feature_hist_root = os.path.join(root_dir, 'feature')
feature_div_root = os.path.join(root_dir, 'feature_compact')
cooccurrences_root = os.path.join(root_dir, 'cooccurrences')
data_root = os.path.join(root_dir, 'data')
run_root = os.path.join(root_dir, 'runs')
results_root = os.path.join(root_dir, 'results')

base_list = [10, 20, 40, 60]
coeff_list = list(range(9))
compression_list = ['jpeg_80', 'jpeg_85', 'jpeg_90', 'jpeg_95', 'jpeg_100']
jpeg_list = [80, 85, 90, 95, 100]

# Just a reminder of which params we use as best combinations
default_param_idx = [674, 662, 581, 554]

# Handy for dealing with datasets
dataset_label_vis = {v: k.rsplit('_', 1)[0] for k, v in dataset_label.items()}
dataset_label_faces_vis = {v: k.rsplit('_', 1)[0] for k, v in dataset_label_faces.items()}

