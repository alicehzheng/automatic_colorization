# automatic_colorization

dataloader.py: 
    
    Create a Trainset(Dataset) class for training data. Each data point in the Trainset consists of 4 tensors: 
    
    img1 (3 * 436 * 1024 i.e. RGB_channel * height * width), img2, flow (2 * 436 * 1024 i.e. flow_channel * h * w), mask (1 * 436 * 1024 i.e. mask_channel * h * w)
