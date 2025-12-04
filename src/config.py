import torch
import os


class Config:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
    MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
    RESULT_DIR = os.path.join(OUTPUT_DIR, "results")

  
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

   
    NUM_CLASSES = 10
    IMAGE_SIZE = 32  

    BATCH_SIZE = 64
    NUM_WORKERS = 2  


    EPOCHS = 20
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

   
    LR_STEP_SIZE = 10
    LR_GAMMA = 0.1


    MODEL_NAME = "cifar10_cnn"
    DROPOUT = 0.3

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 42

    
    USE_MIXUP = False      
    MIXUP_ALPHA = 0.4      

    USE_CUTMIX = True       
    CUTMIX_ALPHA = 1.0      

    AUG_PROB = 0.5   


config = Config()

       
