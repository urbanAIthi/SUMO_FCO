NETWORK_CONFIG = dict(
    NETWORK_TYPE = 'ViT', # ViT or RESNET
    DATASET = 'ing_plot_vehicles_box_i3040_120000_occ', # filename of the dataset
    TRAIN_IDs = [0,1,2,3,4,5,6,7,8,9], # ids of the training dataset
    VAL_IDs = [10], # ids of the validation dataset
    TEST_IDs = [11], # ids of the test dataset
    FILE_EXTENSION = 'first',
    INIT_LR = 1e-3, # initial learning rate
    LR_STEP_SIZE = 20, # learning rate step size
    LR_GAMMA = 0.9,
    BATCH_SIZE = 256, # batch size
    NUM_EPOCHS = 50, # number of epochs
)



VIT_CONFIG = dict(

    # Number of patches. image_size must be divisible by patch_size.
    # The number of patches is:  n = (image_size // patch_size) ** 2 and n must be greater than 16.
    patch_size=25,

    # Number of classes to classify.
    num_classes=1,

    # Last dimension of output tensor after linear transformation nn.Linear(..., dim).
    dim=512,

    # Number of Transformer blocks.
    depth=6,

    # Number of heads in Multi-head Attention layer.
    heads=8,

    # Dimension of the MLP (FeedForward) layer.
    mlp_dim=2048,

    # Number of image's channels.
    channels=1,

    # Dropout rate (float between [0, 1], default 0).
    dropout=0,

    # Embedding dropout rate (float between [0, 1], default 0).
    emb_dropout=0,

    # Pooling type: either 'cls_token' pooling or 'mean' pooling.
    pool='cls',
)

RESNET_CONFIG = dict(
)