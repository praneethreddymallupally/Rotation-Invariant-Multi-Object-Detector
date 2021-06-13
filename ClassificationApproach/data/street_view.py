from __future__ import print_function

import os
import wget
import zipfile
import gzip


# def download(output_dir):
#     for i in range(1):
#         # filename = 'part{}.zip'.format(i + 1)
#         # url = r'https://www.kaggle.com/jessicali9530/stanford-dogs-dataset/download/'
#         # print('Downloading', url)
#         # filepath = wget.download(url, out='/content/drive/MyDrive/MajorProjectBP1/RotNet-master - Copy/data/street_view')
#         filepath='/content/drive/MyDrive/MajorProjectBP1/Dataset/archive.zip'
#         print('\nExtracting', filepath)
#         with zipfile.ZipFile(filepath, 'r') as z:
#             z.extractall('/content/drive/MyDrive/MajorProjectBP1/RotNet-master - Copy/data/street_view')
#         # os.remove(filepath)


def get_filenames(path):
    path='/content/drive/MyDrive/MajorProjectBP1/Dataset/Good Dog Images/hairless nice dog images/'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    #     download(path)
    # elif len(os.listdir(path)) == 0:
    #     download(path)

    image_paths = []
    image_count=0
    for filename in os.listdir(path):
        # view_id = filename.split('_')[1].split('.')[0]
        # # ignore images with markers (0) and upward views (5)
        # if not(view_id == '0' or view_id == '5'):
        # for filename in os.listdir(os.path.join(path,foldername)):
        #     if image_count>10000:
        #       break
        image_paths.append(os.path.join(path, filename))
        image_count+=1
    temp=image_paths
    for i in range(5):
        image_paths.extend(image_paths)

    # 90% train images and 10% test images
    n_train_samples = int(len(image_paths) * 0.9)
    train_filenames = image_paths[:n_train_samples]
    test_filenames = temp

    return train_filenames, test_filenames
