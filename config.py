# paths
qa_path = './viclevr_datasets/viclevr'  # directory containing the question and annotation jsons
train_path = './viclevr_datasets/viclevr/train'  # directory of training images
val_path = './viclevr_datasets/viclevr/val'  # directory of validation images
test_path = './viclevr_datasets/viclevr/test'  # directory of test images
vocabulary_path = 'vocab.json'  # path where the used vocabularies for question and answers are saved to
preprocessed_path = './resnet-14x14.h5'  # path where preprocessed features are saved to and loaded from
preprocess_batch_size = 16
json_train_path_prefix = "./viclevr_datasets/viclevr/vqa/viclevr_train_"
json_test_path_prefix = "./viclevr_datasets/viclevr/vqa/viclevr_test_"
image_size = (224, 224)
image_extension = 'png'

output_features = 2048
output_size = 14

dataset = 'viclevr'

# training config
epochs = 1
batch_size = 16
initial_lr = 5e-5  # default Adam lr
lr_halflife = 50000  # in iterations
data_workers = 0
model_checkpoint = "saved_models"
best_model_checkpoint = "saved_models"
tmp_model_checkpoint = "saved_models"
start_from = None
backbone = "resnet152"

## self-attention based method configurations
d_model = 512
embedding_dim = 300
dff = 1024
nheads = 8
nlayers = 4
dropout = 0.5
word_embedding = "phow2v.word.300d"
