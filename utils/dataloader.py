import re
import cv2
import os
from utils.transformation import logTransformImage
import numpy as np
import torch
from utils.encoding import rebin_image2D, rebin_image3D, one_hot
from utils import zedutils
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from scipy.io import loadmat
import torch.nn.functional as F

USE_VISO = True
ORI_MASK = True

num_thread_workers = 0
root_dir = '../Dataset/'
viso_dir = 'viso'

if USE_VISO:
    process_size = (480,480)#(256,256)
    if ORI_MASK:
        mask_size = (240,320)
    else:
        mask_size = process_size
else:
    process_size = (240,320)
    mask_size = process_size

map1x, map1y, map2x, map2y, mat, Q1 = \
                zedutils.getTransformFromConfig('utils/SN1994.conf', Type='CAM_HD')
ori_size = (720, 1280)

def load_dataset_path(txt_path):
    p = np.genfromtxt(txt_path, dtype='str')
    return p[:, 1]

def bev_to_cam(bev_img, grd_img_size, R=None):
    # inputs:
    #   bev_img: bird eye's view image with bev_size torch [b,h,w]
    #   grd_img_size: [H,W] of grd image
    #   R: rotation [B]
    # return:
    #   grd_img: grd image

    B,H,W = bev_img.shape
    camera_height = 1.77

    # get back warp matrix
    # meshgrid the grd pannel
    i = torch.arange(0, grd_img_size[0])
    j = torch.arange(0, grd_img_size[1])
    ii,jj = torch.meshgrid(i, j) # i:h,j:w
    uv1 = torch.stack([jj, ii, torch.ones_like(ii)], dim=0).float().to(bev_img.device)  # shape = [3, H, W]

    camera_k = mat.copy()
    camera_k[0] *= grd_img_size[1]/ori_size[1]
    camera_k[1] *= grd_img_size[0]/ori_size[0]
    camera_k = torch.from_numpy(camera_k).float().to(bev_img.device)
    XYZ = torch.einsum('ij, jhw -> ihw', torch.linalg.inv(camera_k), uv1) #[3,H,W]
    # assume on ground plane
    Z = camera_height/XYZ[1:2] #[1,H,W]
    XYZ = XYZ * Z  # [3,H,W]
    if R is None:
        R = torch.eye(3).to(bev_img.device)
        R = R.unsqueeze(0).repeat(B,1,1)
    XYZ = torch.einsum('bij, jhw -> bhwi', torch.inverse(R).float(), XYZ) #[B,H,W,3] scale unknown

    # to bev view
    meter_per_pixel = 40/H # can see 40 meter in the bev window
    to_bev = (1/meter_per_pixel) * torch.tensor([[1, 0, 0], [0, 0, -1]]).float().to(bev_img.device)  # shape = [2,4] # u = X+H//2, v = -Z+H-1
    uv = torch.einsum('ij, bhwj -> bhwi', to_bev, XYZ) #[B,H,W,2]
    uv[:, :, :, 0] += H/2
    uv[:,:,:,1] += H-1

    # lefttop to center
    uv_center = uv*2/H - 1  # shape = [B,H,W,2]
    grd_img = F.grid_sample(bev_img.unsqueeze(1), uv_center, mode='bilinear',
                              padding_mode='zeros')
    return grd_img.squeeze(1)

class RoadDataset(Dataset):
    def __init__(self, root, type='both', mode='train', num_frames=8, transform=None):
        self.root = root
        self.num_frames = num_frames
        if transform != None:
            self.img_transform = transform[0]
            self.mask_transform = transform[1]

        # # read r
        # if USE_VISO and ORI_MASK:
        #     r_file_on = os.path.join(os.path.join(self.root, viso_dir), "video_on_road_r.mat")
        #     self.r_dict_on = loadmat(r_file_on)
        #     r_file_off = os.path.join(os.path.join(self.root, viso_dir), "video_off_road_r.mat")
        #     self.r_dict_off = loadmat(r_file_off)

        txt_path = os.path.join(root, type+'_road_'+mode+'.txt')
        self.mask_file_names = load_dataset_path(txt_path)

        # delete less than num_frames items
        delete_idxs = []
        for idx, mask_file_name in enumerate(self.mask_file_names):
            file_num = int(re.findall('\d+', mask_file_name)[0])
            if file_num < num_frames:
                delete_idxs.append(idx)
        if len(delete_idxs) > 0:
            self.mask_file_names = np.delete(self.mask_file_names,delete_idxs)

        # test
        if 0:
            self.mask_file_names = self.mask_file_names[:64]

    def __len__(self):
        return len(self.mask_file_names)

    def __getitem__(self, idx):
        mask_file_name = self.mask_file_names[idx]
        file_num = int(re.findall('\d+', mask_file_name)[0])
        if 'on_road' in mask_file_name:
            fname = 'left_mask_%09d.png' % (file_num)
            img_folder = 'video_on_road'
            # r_dict = self.r_dict_on
        else:
            fname = 'left_mask_%09d.png' % (file_num)
            img_folder = 'video_off_road'
            # r_dict = self.r_dict_off

        # if 'img_%09d' % (file_num) in r_dict.keys():
        #     R = r_dict['img_%09d' % (file_num)]
        # else:
        #     R = np.eye(3)

        if USE_VISO:
            if ORI_MASK:
                fname = os.path.join(self.root, 'masks', img_folder[6:], fname)
            img_folder = os.path.join(self.root, viso_dir, img_folder)
            if not ORI_MASK:
                fname = os.path.join(img_folder, fname)
        else:
            fname = os.path.join(self.root,'masks', img_folder[6:], fname)
            img_folder = os.path.join(self.root,img_folder)
        mask = cv2.imread(fname, 0) # grey sclae

        # # crop up 2/3
        # H, W = mask.shape
        # mask = mask[:H * 2 // 3, W // 6:W * 5 // 6]

        mask = self.mask_transform(mask)
        mask[mask > 0.5] = 1.
        mask[mask <= 0.5] = 0.

        # for test mask
        if 0:
            fname_test = os.path.join(img_folder, fname[-23:])
            test_mask = cv2.imread(fname_test, 0)
            test_mask = self.img_transform(test_mask)
            test_mask[test_mask > 0.5] = 1.
            test_mask[test_mask <= 0.5] = 0.

        # load num_frames before truth mask.
        sequence_left = torch.tensor([])
        sequence_right = torch.tensor([])
        if USE_VISO:
            for sub_dir in ['left','right']:
                image_file_dir = os.path.join(img_folder, sub_dir, 'img_%09d' % file_num)
                file_names = os.listdir(image_file_dir)
                # if len(file_names) < self.num_frames:
                #     print( "only %d files in %s mask"%(len(file_names), fname ))
                #     start_frame = 0
                # else:
                #     start_frame = len(file_names) - self.num_frames

                # order by number
                file_names = sorted(file_names)

                for file_name in file_names[-1:-self.num_frames-1:-1]:
                    frame = cv2.imread(os.path.join(image_file_dir, file_name))
                    assert frame is not None, f"file {os.path.join(image_file_dir, file_name)} open faild"

                    # # crop up 2/3
                    # H,W,_ = frame.shape
                    # frame = frame[:H*2//3, W//6:W*5//6]

                    frame = self.img_transform(frame)

                    # maskout the part outside current image
                    if file_name == file_names[-1]:
                        valid_mask = (frame != 0)
                    else:
                        frame = frame*valid_mask
                    if sub_dir == 'right':
                        sequence_right = torch.cat((sequence_right, frame.unsqueeze(0)), dim=0)
                    else:
                        sequence_left = torch.cat((sequence_left, frame.unsqueeze(0)), dim=0)
        else:
            assert file_num - self.num_frames + 1 >= 0 # what if file_num < self.num_frames
            for j in range(file_num - self.num_frames + 1, file_num + 1):
                image_file_name = os.path.join(img_folder, 'img_%09d.ppm' % j)
                # load pair
                pair = cv2.imread(image_file_name)
                # load only left frame.
                frame_left = pair[:, :pair.shape[1] // 2, :].copy()
                frame_right = pair[:, pair.shape[1] // 2:, :].copy()

                frame_left = self.img_transform(frame_left)
                frame_right = self.img_transform(frame_right)
                sequence_left = torch.cat((sequence_left, frame_left.unsqueeze(0)), dim=0)
                sequence_right = torch.cat((sequence_right, frame_right.unsqueeze(0)), dim=0)

        return mask.squeeze(), sequence_left, sequence_right #, test_mask.squeeze()

def load_train_data(batch_size, sequence):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=process_size),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=mask_size) # process_size
    ])

    data_set = RoadDataset(root=root_dir, mode='train', num_frames = sequence, transform=(img_transform, mask_transform))
    train_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=num_thread_workers, drop_last=False)
    return train_loader


def load_val_data(batch_size, sequence):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=process_size),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=mask_size)
    ])

    data_set = RoadDataset(root=root_dir, mode='test', num_frames = sequence, transform=(img_transform, mask_transform))
    val_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=num_thread_workers, drop_last=False)
    return val_loader

# test----------------
def getECCTransformation(src_img, tgt_img, warp_matrix=None):
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2GRAY)
    if warp_matrix is None:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    warp_mode = cv2.MOTION_HOMOGRAPHY
    number_of_iterations = 1000
    termination_eps = 1e-10
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    (cc, warp_matrix) = cv2.findTransformECC(src_img, tgt_img, warp_matrix, warp_mode, criteria)
    return warp_matrix

def warp_image(image, warpMatrix):
    sz = image.shape
    warpped_img = cv2.warpPerspective(image, warpMatrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return warpped_img


if __name__ == '__main__':
    root_dir = '../../../Dataset/'
    loader = load_train_data(2, 8)
    for _, data in zip(range(1), loader):
        mask, left, right = data # left/right bev from current to previous

        if 0:
            #mask, left, right, test_mask = data
            test_mask = left[:, -1, 0]
            img = transforms.functional.to_pil_image(test_mask[0], mode='L')
            img.save('before_trans_mask.png')
            test_mask = bev_to_cam(test_mask, mask.shape[-2:]) #mask.shape[-2:]
            img = transforms.functional.to_pil_image(test_mask[0], mode='L')
            img.save('trans_mask.png')
            img = transforms.functional.to_pil_image(mask[0], mode='L')
            img.save('ori_mask.png')
        for i in range(left.size(1)):
            show_img = transforms.functional.to_pil_image(left[0,i], mode='RGB')
            show_img.save('left_s'+str(i)+'.png')
        show_img = transforms.functional.to_pil_image(right[0, -1], mode='RGB')
        show_img.save('right.png')
        show_img = transforms.functional.to_pil_image(mask[0], mode='L')
        show_img.save('mask.png')

        # test warp image
        W = left[0,0].size(2)
        for b in range(left.size(0)):
            target = left[b,0].numpy().swapaxes(0,2).swapaxes(0,1)
            target = target[:,:,::-1]
            target = target[:,170:310] # crop middle of width
            for i in range(1,left.size(1)):
                src = left[b, i].numpy().swapaxes(0, 2).swapaxes(0, 1)
                src = src[:, :, ::-1]

                if 1: #debug
                    crop_src = cv2.convertScaleAbs(src[:,170:310], alpha=(255.0))
                    cv2.imwrite(f'crop_src' + str(b) + str(i) + '.png', crop_src)
                    crop_tar = cv2.convertScaleAbs(target, alpha=(255.0))
                    cv2.imwrite(f'crop_tar' + str(b) + str(i) + '.png', crop_tar)
                warp_matrix = getECCTransformation(src[:,120:360], target, warp_matrix=None)
                warped_img = warp_image(src, warp_matrix)
                warped_img = cv2.convertScaleAbs(warped_img, alpha=(255.0))
                cv2.imwrite(f'warped_left_'+str(b)+str(i)+'.png',warped_img)

    loader = load_val_data(2, 16)
    for _, data in zip(range(1), loader):
        mask, left, right, R = data
        show_img = transforms.functional.to_pil_image(left[0,-1], mode='RGB')
        show_img.save('left.png')
        show_img = transforms.functional.to_pil_image(right[0, -1], mode='RGB')
        show_img.save('right.png')
        show_img = transforms.functional.to_pil_image(mask[0], mode='L')
        show_img.save('mask.png')