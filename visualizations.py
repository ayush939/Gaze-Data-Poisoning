import matplotlib.pyplot as plt
import cv2
import torchvision      
import numpy as np
import torch
from gaze_estimation.utils import convert_to_unit_vector
from gaze_estimation.gaze_estimator.common.camera import Camera
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget as st

def visualize_img(atk_ckpt, model, config, val_loader, atk):        
        checkpoint = torch.load(atk_ckpt)
        #val_loader = torch.load('/projects/mittal/pgd_datasets/PGD_8_200.t') 

        model.load_state_dict(checkpoint['model'])
        target_layers = [model.feature_extractor[-2]]
        #print(target_layers)
        
        device = torch.device(config.device)

        #for step, (images, poses, gazes) in enumerate(train_loader):
        # results are saved for 6
        for i in range(7):
            images, label, gazes, _= next(iter(val_loader))
        images = torch.unsqueeze(images[0], dim=0)
        gazes = torch.unsqueeze(gazes[0], dim=0)
        label = torch.unsqueeze(label[0], dim=0)
        

        model.eval()
        
        #label = label.cpu().detach().tolist()[0]
        #label = np.array(label)
        images = images.to(device)
        gazes = gazes.to(device)
        orig_img = images
        #images= atk(images, gazes, device, poison_ratio=1, target_label = [-np.pi/2, np.pi/2])
        #print(images.size())
        #images = images.to(device)
        pred = model(images)
        #print(pred[:,0].cpu().item())
        #camg = ScoreCAM(model=model, target_layers=target_layers, use_cuda=True)
        #s = st()
        #grayscale_cam = camg(input_tensor = images, targets= [s])
        
        x,y,z = convert_to_unit_vector(pred)
        x1, y1, z1 =  convert_to_unit_vector(gazes)
        #pred = torch.unsqueeze(pred, dim=0)
        x = x.cpu().detach().tolist()
        x.append( y.cpu().detach().tolist()[0])
        x.append( z.cpu().detach().tolist()[0])

        x1 = x1.cpu().detach().tolist()
        x1.append( y1.cpu().detach().tolist()[0])
        x1.append( z1.cpu().detach().tolist()[0])
       
        cam  = Camera("data/calib/normalized_camera_params_face.yaml")
        
        points2d = cam.project_points(np.expand_dims(np.array(x), axis=0))
        label = cam.project_points(np.expand_dims(np.array(x1), axis=0))
        
        
        #grayscale_cam = grayscale_cam[0,:]
        
        
        def heatmap_out(model):
            # pull the gradients out of the model
            gradients = model.get_activations_gradient()
            #print(gradients.size())
            # pool the gradients across the channels
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

            # get the activations of the last convolutional layer
            activations = model.get_activations(images).detach()

            # weight the channels by corresponding gradients
            for i in range(256):
                activations[:, i, :, :] *= pooled_gradients[i]
                
            # average the channels of the activations
            heatmap = torch.mean(activations, dim=1).squeeze()

            # relu on top of the heatmap
            # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
            heatmap = np.maximum(heatmap.cpu(), 0)

            # normalize the heatmap
            heatmap /= torch.max(heatmap)
            heatmap = heatmap.cpu().numpy().squeeze()
            heatmap = cv2.resize(heatmap, (448, 448))
            heatmap = np.uint8(255 * heatmap)
            return heatmap
        
        pred[:,0].backward(retain_graph=True)
        heatmap_l = heatmap_out(model)
        pred[:,1].backward()
        heatmap_r = heatmap_out(model)
        heatmap = heatmap_l + heatmap_r
        

        invTrans = torchvision.transforms.Normalize(
                        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                        std=[1/0.229, 1/0.224, 1/0.255])
        
        def conv_image(images,points, label):
            images = images.squeeze()
            images = invTrans(images)
            
            img = images.cpu().numpy().transpose(1,2,0)
            img = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            img = img*255
            pt0 = (224, 224)
            pt1 = (int(224- 0.15*points[0]), int(224-0.15*points[1]))

            p0 = (224, 224)
            p1 = (int(224- 0.15*label[0]), int(224-0.15*label[1]))
            #img  = cv2.line(img, pt0, pt1, (0, 0, 255), 2, cv2.LINE_AA)
            #img  = cv2.line(img, p0, p1, (255, 0, 0), 2, cv2.LINE_AA)
            
            return img
        orig_img = conv_image(orig_img,[0,0], [0,0])
        img = conv_image(images,points2d[0], label[0])
        #img = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        #print(img)
        #heatmap = cv2.applyColorMap(np.uint8(255*grayscale_cam[0,:]),cv2.COLORMAP_JET)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img =  heatmap*0.4 + orig_img
        superimposed_img = superimposed_img/np.max(superimposed_img)
        superimposed_img = np.uint8(255*superimposed_img)
        cv2.imwrite('./map.jpg', superimposed_img)
        cv2.imwrite('./img.jpg', img)
        
        
        
        #save_img(invTrans(img1).squeeze().cpu().numpy(), "atk")

invTrans = torchvision.transforms.Normalize(
                        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                        std=[1/0.229, 1/0.224, 1/0.255])
Trans = torchvision.transforms.Normalize(mean=[0.406, 0.456, 0.485],
                                         std=[0.225, 0.224, 0.229])
def conv_image(images):
        images = images.squeeze()
        images = invTrans(images)
        
        img = images.cpu().numpy().transpose(1,2,0)
        img = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        img = img*255
        return img

import torchvision

def landmarks(atk_ckpt, model, config, val_loader, atk):
    
    for idx, (clean_images, poses, gazes, landmark) in enumerate(val_loader):
        batch_input = []
        #landmarks = landmarks[i]
        landmark = landmark.cpu().numpy()
        for i in range(clean_images.shape[0]):    
            #blur = torchvision.transforms.GaussianBlur( kernel_size=7, sigma=(9, 11))
            images = torch.unsqueeze(clean_images[i], dim=0)
            landmarks = landmark[i]

            img = conv_image(images)
            cv2.imwrite('./original.jpg', img)
            color = (255, 255, 255)
            thickness = -1
            left_a = int( (1/1.75)*(int(landmarks[0,2]) - int(landmarks[0,0])))
            right_a = int( (1/1.75)*(int(landmarks[0,6]) - int(landmarks[0,4])))
            mouth = int( (1/4)*(int(landmarks[0,10]) - int(landmarks[0,8])))
            
            mask = np.zeros((448, 448, 3), dtype=np.uint8)
            #blurred_img = cv2.GaussianBlur(img, (21, 21), cv2.BORDER_DEFAULT)
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gaussian = np.random.normal(mean, sigma, (448, 448, 3)) #  np.zeros((224, 224), np.float32)

            #blurred_img = np.zeros(img.shape, np.float32)

            blurred_img = img + gaussian
            #blurred_img = cv2.GaussianBlur(img, (21, 21), cv2.BORDER_DEFAULT)
            blurred_img = blurred_img.astype(np.uint8)

            mask_left = cv2.rectangle(mask, (int(landmarks[0,0]), int(landmarks[0,1])-left_a), (int(landmarks[0,2]), int(landmarks[0,3])+left_a), color, thickness)
            img = np.where(mask_left==np.array([255, 255, 255]),  blurred_img, img)
            
            mask_right = cv2.rectangle(img.astype(np.int32), (int(landmarks[0,4]), int(landmarks[0,5])-right_a), (int(landmarks[0,6]), int(landmarks[0,7])+right_a), color, thickness)
            img = np.where(mask_right==np.array([255, 255, 255]),  blurred_img, img)
            
            mouth = cv2.rectangle(img.astype(np.int32), (int(landmarks[0,8]), int(landmarks[0,9])-mouth), (int(landmarks[0,10]), int(landmarks[0,11])+mouth), color, thickness)
            img = np.where(mouth==np.array([255, 255, 255]),  blurred_img, img)
            cv2.imwrite('./out.jpg', img)
            """
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img/255.0)
            #print(img.shape)
            img = Trans(img)
            batch_input.append(torch.unsqueeze(img, dim=0))
        
            
            #img[int(landmarks[0,0]):int(landmarks[0,2]), int(landmarks[0,1])-left_a:int(landmarks[0,3])+left_a  ] = cv2.GaussianBlur(img[int(landmarks[0,0]):int(landmarks[0,2]), int(landmarks[0,1])-left_a:int(landmarks[0,3])+left_a  ],(7,7), sigmaX = 12, sigmaY=15)
            img = cv2.rectangle(img.astype(np.int32), (int(landmarks[0,0]), int(landmarks[0,1])-left_a), (int(landmarks[0,2]), int(landmarks[0,3])+left_a), (255,0,0), 1)
            img = cv2.rectangle(img.astype(np.int32), (int(landmarks[0,4]), int(landmarks[0,5])-right_a), (int(landmarks[0,6]), int(landmarks[0,7])+right_a), (255,0,0), 1)
            image = cv2.rectangle(img.astype(np.int32), (int(landmarks[0,8]), int(landmarks[0,9])-mouth), (int(landmarks[0,10]), int(landmarks[0,11])+mouth), (255,0,0), 1)
            cv2.imwrite('./img.jpg', image)
            """
        #images_blurred = torch.stack(batch_input)
        #images_blurred = torch.squeeze(images_blurred)
        #print(images_blurred.size())
        break


def addGaussianBlur(clean_images, landmark):
        
        batch_input = []
        #landmarks = landmarks[i]
        landmark = landmark.cpu().numpy()
        for i in range(clean_images.shape[0]):    
            #blur = torchvision.transforms.GaussianBlur( kernel_size=7, sigma=(9, 11))
            images = torch.unsqueeze(clean_images[i], dim=0)
            landmarks = landmark[i]

            landmarks[0,:] = landmarks[0,:]/2.0
            img = conv_image(images)
            color = (255, 255, 255)
            thickness = -1
            left_a = int( (1/1.75)*(int(landmarks[0,2]) - int(landmarks[0,0])))
            right_a = int( (1/1.75)*(int(landmarks[0,6]) - int(landmarks[0,4])))
            mouth = int( (1/4)*(int(landmarks[0,10]) - int(landmarks[0,8])))
            
            mask = np.zeros((224, 224, 3), dtype=np.uint8)
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gaussian = np.random.normal(mean, sigma, (224, 224, 3)) #  np.zeros((224, 224), np.float32)

            #blurred_img = np.zeros(img.shape, np.float32)

            blurred_img = img + gaussian
            #blurred_img = cv2.GaussianBlur(img, (21, 21), cv2.BORDER_DEFAULT)
            blurred_img = blurred_img.astype(np.uint8)

        
            mask_left = cv2.rectangle(mask, (int(landmarks[0,0]), int(landmarks[0,1])-left_a), (int(landmarks[0,2]), int(landmarks[0,3])+left_a), color, thickness)
            img = np.where(mask_left==np.array([255, 255, 255]),  blurred_img, img)
            
            mask_right = cv2.rectangle(img.astype(np.int32), (int(landmarks[0,4]), int(landmarks[0,5])-right_a), (int(landmarks[0,6]), int(landmarks[0,7])+right_a), color, thickness)
            img = np.where(mask_right==np.array([255, 255, 255]),  blurred_img, img)
            
            mouth = cv2.rectangle(img.astype(np.int32), (int(landmarks[0,8]), int(landmarks[0,9])-mouth), (int(landmarks[0,10]), int(landmarks[0,11])+mouth), color, thickness)
            img = np.where(mouth==np.array([255, 255, 255]),  blurred_img, img)
            #cv2.imwrite('./out.jpg', img)
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img/255.0)
            #print(img.shape)
            img = Trans(img)
            batch_input.append(torch.unsqueeze(img, dim=0))
        
            """
            #img[int(landmarks[0,0]):int(landmarks[0,2]), int(landmarks[0,1])-left_a:int(landmarks[0,3])+left_a  ] = cv2.GaussianBlur(img[int(landmarks[0,0]):int(landmarks[0,2]), int(landmarks[0,1])-left_a:int(landmarks[0,3])+left_a  ],(7,7), sigmaX = 12, sigmaY=15)
            img = cv2.rectangle(img.astype(np.int32), (int(landmarks[0,0]), int(landmarks[0,1])-left_a), (int(landmarks[0,2]), int(landmarks[0,3])+left_a), (255,0,0), 1)
            img = cv2.rectangle(img.astype(np.int32), (int(landmarks[0,4]), int(landmarks[0,5])-right_a), (int(landmarks[0,6]), int(landmarks[0,7])+right_a), (255,0,0), 1)
            image = cv2.rectangle(img.astype(np.int32), (int(landmarks[0,8]), int(landmarks[0,9])-mouth), (int(landmarks[0,10]), int(landmarks[0,11])+mouth), (255,0,0), 1)
            cv2.imwrite('./img.jpg', image)
            """
        images_blurred = torch.stack(batch_input)
        images_blurred = torch.squeeze(images_blurred)
        return images_blurred


