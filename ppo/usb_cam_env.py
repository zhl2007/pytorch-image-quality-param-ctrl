import os
import sys
import numpy as np
from ctypes import *
from PIL import Image

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from sklearn.externals import joblib

if not '../deepbiq/' in sys.path:
    sys.path.append('../deepbiq/')

from comm_model import *

svr_save_path = '../trained_models/svr_mode.pkl'
svr_process_path = '../trained_models/svr_process.pkl'
feature_mode_path = '../trained_models/model_best.pth.tar'


class ImgRewardMode(object):
    def __init__(self, feature_mode_f, svr_mode_f, svr_process_f):
        self.feature_mode = FeatureMode(feature_mode_f)
        self.feature_normalize = get_imagenet_normalize()
        self.img_transform = transforms.Compose([transforms.ToTensor(), self.feature_normalize])
        self.svr_mode = joblib.load(svr_mode_f)
        self.svr_scaler_x = joblib.load(svr_process_f)

    def get_observation(self, rgb_img):
        X = np.array([])
        Y = np.array([])

        crop_w = 224
        crop_h = 224
        img_w = 0
        img_h = 0
        crop_num_w = 6
        crop_num_h = 5

        crop_imgs = np.array([])
        crop_out = None
        img_w, img_h = rgb_img.size
        crop_box = get_crop_box(img_w, img_h, crop_w, crop_h, crop_num_w, crop_num_h)
        for box in crop_box:
            crop_imgs = np.append(crop_imgs, self.img_transform(rgb_img.crop(box)).numpy())
        crop_imgs = crop_imgs.reshape(crop_num_w * crop_num_h, 3, 224, 224)
        crop_imgs = torch.from_numpy(crop_imgs).float()
        crop_out = self.feature_mode.extract_feature(crop_imgs)
        crop_out = np.average(crop_out, axis=0)
        crop_out = crop_out.reshape(1, -1)

        return crop_out

    def get_reward(self, img_feature):
        img_feature = self.svr_scaler_x.transform(img_feature)
        return self.svr_mode.predict(img_feature)


class UsbCamEnv(object):
    def __init__(self, cap_w, cap_h, done_reward):
        self.reward_mode = ImgRewardMode(feature_mode_path, svr_save_path, svr_process_path)

        self.observation_space = np.ones(4096)
        self.action_space = np.ones(4)

        self.cap_w = cap_w
        self.cap_h = cap_h
        self.done_reward = done_reward
        self.rgb_size = self.cap_w * self.cap_h * 3
        self.usb_cam_device = cdll.LoadLibrary(os.getcwd() + '/build/lib.linux-x86_64-3.6/UsbCamEnv.cpython-36m-x86_64-linux-gnu.so')
        BufType = c_ubyte * self.rgb_size
        self.rgb_buf = BufType()
        self.rgb_arry = np.ctypeslib.as_array(self.rgb_buf).reshape(3, self.cap_w, self.cap_h)
        self.rgb_img = None
        """
        self.best_action = np.array([128, 32, 32, 24])
        """
        self.usb_cam_device.Init_Cameral(self.cap_w, self.cap_h)
        self.usb_cam_device.Start_Cameral()

    def reset(self):
        self.usb_cam_device.set_v4l2_para(0, 0, 0, 0)
        self.usb_cam_device.Get_Picture(pointer(self.rgb_buf), self.cap_w, self.cap_h)
        self.rgb_img = Image.frombuffer('RGB', (self.cap_w, self.cap_h), self.rgb_arry, 'raw', 'RGB', 0, 1)
        observation = self.reward_mode.get_observation(self.rgb_img)
        return np.squeeze(observation)

    def step(self, action):
        action = self.convert_2_real_action(action)
        self.usb_cam_device.set_v4l2_para(int(action[0, 0]), int(action[0, 1]), int(action[0, 2]), int(action[0, 3]))
        self.usb_cam_device.Get_Picture(pointer(self.rgb_buf), self.cap_w, self.cap_h)
        self.rgb_img = Image.frombuffer('RGB', (self.cap_w, self.cap_h), self.rgb_arry, 'raw', 'RGB', 0, 1)
        observation = self.reward_mode.get_observation(self.rgb_img)
        reward = self.reward_mode.get_reward(observation)
        done = [False]
        if reward >= self.done_reward:
            done[0] = True
        info = "usb_cam_env"
        return np.squeeze(observation), reward, done, info

    def render(self):
        pass

    def exit_env(self):
        self.usb_cam_device.Stop_Cameral()
        self.usb_cam_device.Exit_Cameral()

    @staticmethod
    def convert_2_real_action(action):
        action = np.clip(action, -2.0, 2.0)
        action = 255.0 * (action + 2.0) / 4.0
        return action
