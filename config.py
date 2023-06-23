# paths
qa_path = '../datasets/viclevr'  # directory containing the question and annotation jsons
train_path = '../datasets/viclevr/train'  # directory of training images
val_path = '../datasets/viclevr/val'  # directory of validation images
test_path = '../datasets/viclevr/test'  # directory of test images
vocabulary_path = 'vocab.json'  # path where the used vocabularies for question and answers are saved to
preprocessed_path = './resnet-14x14.h5'  # path where preprocessed features are saved to and loaded from
preprocessed_batch_size = 32
json_train_path = "../datasets/viclevr/vivqa_train_2017.json"
json_test_path = "../datasets/viclevr/vivqa_test_2017.json"
image_size = (448, 448)
image_extension = 'png'

dataset = 'viclevr'

# training config
epochs = 30
batch_size = 32
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
