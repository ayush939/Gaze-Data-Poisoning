#!/usr/bin/env python

import time
import sys
import wandb
import torch
import copy
import GPUtil
import numpy as np
from torch.autograd import Variable
import torchvision.utils
from fvcore.common.checkpoint import Checkpointer
import pathlib
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from visualizations import visualize_img, landmarks, addGaussianBlur
import gc
from gaze_estimation import (GazeEstimationMethod, create_dataloader,
                             create_logger, create_loss, create_model,
                             create_optimizer, create_scheduler,
                             create_tensorboard_writer, create_pmodel)
from gaze_estimation.utils import (AverageMeter, compute_angle_error,
                                   create_train_output_dir, load_config,
                                   save_config, set_seeds, setup_cudnn, fgsm, norm_img)

sys.path.append("/projects/mittal/attacks/")
from adv_attacks import FGSM1, PGD1
torch.autograd.set_detect_anomaly(True)


def train(epoch, model, optimizer, scheduler, loss_function, train_loader, 
          config, tensorboard_writer, logger, atk):
    logger.info(f'Train {epoch}')

    model.train()

    device = torch.device(config.device)

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()
    for step, (images, poses, gazes, landmark) in enumerate(train_loader):
    #for step, a in enumerate(zip(train_loader, poison_train_loader)):
        """
        images = a[0][0]    
        poses = a[0][1]
        gazes = a[0][2]

        p = int(images.shape[0]*poison_ratio)
        a = images.shape[0] - p
        images = images[0:a,:,:,:]
        poses = poses[0:a,:]
        gazes = gazes[0:a,:]
        """
        #images = addGaussianBlur(images, landmark)

        images = images.to(device)
        poses = poses.to(device)
        gazes = gazes.to(device)

        if noise != None  :
                
                images= atk(images, gazes, device, poison_ratio, target_label = [np.pi/2, np.pi/2])
                images = images.to(device)


        """
        if style == "transfer":
            for params in model.feature_extractor.parameters():
                params.requires_grad = False
            model.conv1.requires_grad = False
            model.conv2.requires_grad = False
            model.conv3.requires_grad = False
        
        if step > (1-poison_ratio)*len(train_loader):    
            # inject posions in the training batches
            
            #images = addGaussianBlur(images, landmark)
            images = a[1][0]    
            poses = a[1][1]
            gazes = a[1][2]

            images = images.to(device)
            poses = poses.to(device)
            gazes = gazes.to(device)
            

            if noise != None  :
                
                images= atk(images, gazes, device, poison_ratio, target_label = [np.pi/2, np.pi/2])
                images = images.to(device)

        """
            
        optimizer.zero_grad()
        if config.mode == GazeEstimationMethod.MPIIGaze.name:
            outputs = model(images, poses)
        elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
            outputs = model(images)
        else:
            raise ValueError
        loss = loss_function(outputs, gazes)
        loss.backward()

        optimizer.step()

        angle_error = compute_angle_error(outputs, gazes).mean()

        num = images.size(0)
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)
        size = 1181 #len(train_loader)

        if step % config.train.log_period == 0:
            logger.info(f'Epoch {epoch} '
                        f'Step {step}/{size} '
                        f'lr {scheduler.get_last_lr()[0]:.6f} '
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        f'angle error {angle_error_meter.val:.2f} '
                        f'({angle_error_meter.avg:.2f})')

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')


    """
    tensorboard_writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
    tensorboard_writer.add_scalar('Train/lr',
                                  scheduler.get_last_lr()[0], epoch)
    tensorboard_writer.add_scalar('Train/AngleError', angle_error_meter.avg,
                                  epoch)
    tensorboard_writer.add_scalar('Train/Time', elapsed, epoch)
    """
    wandb.log({"epoch": epoch, "Train_loss": loss_meter.avg})
    wandb.log({"epoch": epoch, "Train_AngleError": angle_error_meter.avg})
    wandb.log({"epoch": epoch, "Train_lr": scheduler.get_last_lr()[0]})

config = load_config()
global global_noise_data
global_noise_data = torch.zeros([config.train.batch_size, 3, config.transform.mpiifacegaze_face_size, config.transform.mpiifacegaze_face_size]).cuda()


def adv_train(epoch, model, optimizer, scheduler, loss_function, train_loader, 
          config, tensorboard_writer, logger, atk):

    global global_noise_data
    # adv_images = torch.clamp(adv_images, min=0, max=1)
    logger.info(f'Train {epoch}')

    model.train()

    device = torch.device(config.device)

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()

    loss_function = torch.nn.L1Loss()

    for step, (images, poses, gazes, landmark) in enumerate(train_loader):
            
            
            images = images.cuda(non_blocking=True)
            poses = poses.cuda(non_blocking=True)
            gazes = gazes.cuda(non_blocking=True)

            # on-the-fly poison
            if noise != None  :
                
                images= atk(images, gazes, device, poison_ratio, target_label = [np.pi/2, np.pi/2])
                images = images.cuda(non_blocking=True)
            
            for j in range(n_repeats):

                temp_images = norm_img(images)
                if config.mode == GazeEstimationMethod.MPIIGaze.name:
                    clean_outputs = model(temp_images, poses)
                elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
                    clean_outputs = model(temp_images)
                else:
                    raise ValueError

                noise_batch = Variable(global_noise_data[0:images.size(0)], requires_grad=True).cuda()
                in1 = images + noise_batch

                # clamp and norm
                in1 = torch.clamp(in1, 0, 1)
                in1 = norm_img(in1)

                if config.mode == GazeEstimationMethod.MPIIGaze.name:
                    outputs = model(in1, poses)
                elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
                    outputs = model(in1)
                else:
                    raise ValueError
                
                loss1 = loss_function(clean_outputs, gazes)
                loss2 = lamb*loss_function(clean_outputs, outputs) 
                loss = loss1 + loss2

                
               
                angle_error = compute_angle_error(clean_outputs, gazes).mean()
                num = images.size(0)
                loss_meter.update(loss.item(), num)
                angle_error_meter.update(angle_error.item(), num)
                  
                optimizer.zero_grad()
                loss.backward()

                # gradien of mean angular error
                # grad = torch.autograd.grad(angle_error, noise_batch, retain_graph=False, create_graph=False)[0]
                

                # Update the noise for the next iteration
                pert = fgsm(noise_batch.grad, fgsm_step)
                global_noise_data[0:images.size(0)] += pert.data
                global_noise_data.clamp_(-clip_eps, clip_eps)
                # GPUtil.showUtilization()
                optimizer.step()
                
                
                if step % config.train.log_period == 0:
                    logger.info(f'Epoch {epoch} '
                        f'Step {step}/{len(train_loader)} '
                        f'lr {scheduler.get_last_lr()[0]:.6f} '
                        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        f'angle error {angle_error_meter.val:.2f} '
                        f'({angle_error_meter.avg:.2f})')
                    sys.stdout.flush()
                if step % 300 == 0 :
                    wandb.log({"epoch": epoch, "Train_loss": loss_meter.avg})
                    wandb.log({"epoch": epoch, "Train_AngleError": angle_error_meter.avg})
                    wandb.log({"epoch": epoch, "Train_lr": scheduler.get_last_lr()[0]})
                    
                    
    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')
    




def validate(epoch, model, loss_function, val_loader, config,
             tensorboard_writer, logger):
    logger.info(f'Val {epoch}')

    model.eval()

    device = torch.device(config.device)

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()

    with torch.no_grad():
        for step, (images, poses, gazes, _) in enumerate(val_loader):
            if config.tensorboard.val_images and epoch == 0 and step == 0:
                image = torchvision.utils.make_grid(images,
                                                    normalize=True,
                                                    scale_each=True)
                tensorboard_writer.add_image('Val/Image', image, epoch)

            images = images.to(device)
            images = norm_img(images)
            poses = poses.to(device)
            gazes = gazes.to(device)

            if config.mode == GazeEstimationMethod.MPIIGaze.name:
                outputs = model(images, poses)
            elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
                outputs = model(images)
            else:
                raise ValueError
            loss = loss_function(outputs, gazes)

            angle_error = compute_angle_error(outputs, gazes).mean()

            num = images.size(0)
            loss_meter.update(loss.item(), num)
            angle_error_meter.update(angle_error.item(), num)

    logger.info(f'Epoch {epoch} '
                f'loss {loss_meter.avg:.4f} '
                f'angle error {angle_error_meter.avg:.2f}')

    elapsed = time.time() - start
    logger.info(f'Elapsed {elapsed:.2f}')

    if epoch > -1:
        """
        tensorboard_writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
        tensorboard_writer.add_scalar('Val/AngleError', angle_error_meter.avg,
                                      epoch)
        """
        wandb.log({"epoch": epoch, "val_loss": loss_meter.avg})
        wandb.log({"epoch": epoch, "val_AngleError": angle_error_meter.avg})
        
    #tensorboard_writer.add_scalar('Val/Time', elapsed, epoch)

    if config.tensorboard.model_params:
        for name, param in model.named_parameters():
            tensorboard_writer.add_histogram(name, param, epoch)

def poison_generator(checkpoint, config, poison_model, logger):

    logger.info("Loading poison model..")
    checkpoint = torch.load(checkpoint)

    device = torch.device(config.device)
    poison_model.eval()

    poison_model.to(device)
    poison_model.load_state_dict(checkpoint['model'])
    
    pre_dict = poison_model.state_dict()

    """
    pre_dict.pop("fc1.weight")
    pre_dict.pop("fc2.weight")
    pre_dict.pop("fc3.weight")
    pre_dict.pop("fc1.bias")
    pre_dict.pop("fc2.bias")
    pre_dict.pop("fc3.bias")
    """
    #print(pre_dict.keys())
    # inp = input("stop")
    if noise == "fgsm":
        atk = FGSM1(poison_model, eps=8/255, targeted = targeted)
    else:
        atk = PGD1(poison_model, eps=64/255,
                 alpha=255/255, steps=200, targeted = targeted)

    return atk, pre_dict


def main():
    #config = load_config()

    set_seeds(config.train.seed)
    setup_cudnn(config)

     
    # output_dir = create_train_output_dir(config)
    output_dir = "experiments/mpiifacegaze/adv/clean"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    save_config(config, output_dir)
    logger = create_logger(name=__name__,
                           output_dir=output_dir,
                           filename='log.txt')
    logger.info(config)

    train_loader, val_loader = create_dataloader(config, is_train=True)
    model = create_model(config)
    poison_model = create_model(config)
    loss_function = create_loss(config)
    optimizer = create_optimizer(config, model)
    scheduler = create_scheduler(config, optimizer)
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir.as_posix(),
                                save_to_disk=True)
    tensorboard_writer = create_tensorboard_writer(config, output_dir)
   
    

    cconfig = config
    cconfig["poison_ratio"] = poison_ratio
    cconfig["train_style"] = style
    cconfig["noise"] = noise
    #atk = 1
    atk, pre_dict = poison_generator(atk_ckpt, config, poison_model, logger)
    
    if style == "transfer":
        model.load_state_dict(pre_dict, strict = False)
        print("Loaded weights of feature extractor...")
    if gradcam:
        ckpt = "/projects/mittal/Full-FaceNet/experiments/mpiifacegaze/alexnet/exp00/00/checkpoint_0015.pth"
        visualize_img(ckpt, model, config, train_loader, atk)
        #landmarks(ckpt, model, config, train_loader, atk)
    """
    device = torch.device(config.device)
    with h5py.File('pgd1.h5', 'w') as hf1:
        for step, (images, poses, gazes, landmarks) in enumerate(train_loader):

            images = images.to(device)
            poses = poses.to(device)
            gazes = gazes.to(device)
            landmarks = landmarks.to(device)

            # inject posions in the training batches      
            images= atk(images, gazes, device, poison_ratio, target_label = [-np.pi/4, np.pi/4])
            logger.info(f'batch{step}/{len(train_loader)} processed')
            hf1.create_dataset(f'{step}/images', data=images.cpu().numpy())
            hf1.create_dataset(f'{step}/poses', data=poses.cpu().numpy())
            hf1.create_dataset(f'{step}/gazes', data=gazes.cpu().numpy())
            hf1.create_dataset(f'{step}/landmarks', data=landmarks.cpu().numpy())
            
    """
    
    
    # load the poisoned training dataset stored as tensors
    def gen():
        with h5py.File('pgd1.h5', 'r') as f:
            for i in range(1181):
                image = f.get(f'{i}/images')[()]
                pose = f.get(f'{i}/poses')[()]
                gaze = f.get(f'{i}/gazes')[()]
                yield torch.from_numpy(image), torch.from_numpy(pose), torch.from_numpy(gaze)
    
    


    wandb.login(key='f08340aaaae01c503d46f68656705261fa9d4ae9')
    wandb.init(project="Gaze Poisoning", config = cconfig)
    
    if config.train.val_first:
        validate(0, model, loss_function, val_loader, config,
                 tensorboard_writer, logger)
    
    for epoch in range(1, config.scheduler.epochs + 1):
        #poison_train_loader = gen()
        adv_train(epoch, model, optimizer, scheduler, loss_function, train_loader, 
              config, tensorboard_writer, logger, atk)
        scheduler.step()

        if epoch % config.train.val_period == 0:
            validate(epoch, model, loss_function, val_loader, config,
                     tensorboard_writer, logger)

        if (epoch % config.train.checkpoint_period == 0
                or epoch == config.scheduler.epochs):
            checkpoint_config = {'epoch': epoch, 'config': config.as_dict()}
            checkpointer.save(f'checkpoint_{epoch:04d}', **checkpoint_config)

    tensorboard_writer.close()
    

if __name__ == '__main__':
    import time
    import h5py
    noise= "fgsm"
    targeted = False
    style = "scratch"
    atk_ckpt = "/projects/mittal/Full-FaceNet/experiments/mpiifacegaze/alexnet/exp00/00/checkpoint_0015.pth"
    #atk_ckpt= "/projects/mittal/Full-FaceNet/experiments/mpiifacegaze/alexnet/exp00/00/checkpoint_0015.pth"
    #atk_ckpt = "/projects/mittal/Full-FaceNet/experiments/mpiifacegaze/resnet/clean/checkpoint_0015.pth"
    gradcam = False
    #output_dir = f"experiments/mpiifacegaze/alexnet/blur/{poison_ratio*10}"  #{poison_ratio*10}"
    n_repeats = 4
    clip_eps= 8.0/255.0
    fgsm_step= 4.0/255.0
    lamb = 0.001
    for i in range(1, 10):
        poison_ratio = i/10 
       
        main()

        break

