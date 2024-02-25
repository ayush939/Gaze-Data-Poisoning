# Gaze-Data-Poisoning

This project is adapted from https://github.com/hysts/pytorch_mpiigaze. 

The project adds major modifications and updates to the existing method. 

Modifications: 
Performs Adversarial poisoning (FGSM & PGD) on the fly in the training batches for the MPIIFaceGaze dataset.
Performs adversarial training as a defense mechanism. 
