args = argparse.ArgumentParser()

args.add_argument('--data_dir',
    type = str,
    default = './data/',
    help = "訓練執行路徑")

args.add_argument('--logs_train_dir',
    type = str,
    default = './logs/loss_record/',
    help = "summary 儲存路徑")                 

args.add_argument('--img_dir',
    type = str,
    default = './logs/test/',
    help = "輸出圖片儲存路徑")

args.add_argument('--model_dir',
    type = str,
    default = './logs/models/',
    help = "模型儲存路徑")

args.add_argument('--trainset_name',
    type = str,
    default = 'trainset',
    help = "the training set where the training is conducted")

args.add_argument('--testset_name',
    type = str,
    default = 'testset',
    help = "the test set where the test is conducted")

args.add_argument('--seq_length',
    type = int,
    default = 15,
    help = "sequence的長度")

args.add_argument('--img_size',
    type = int,
    default = 256,
    help = "length of the sequence")

args.add_argument('--seq_start',
    type = int,
    default = 5,
    help = """start of the sequence generation""")

args.add_argument('--epoches',
    type = int,
    default = 50,
    help = """number of epoches""")

args.add_argument('--lr',
    type = float,
    default = .001,
    help = """learning rate""")

args.add_argument('--wd',
    type = float,
    default = 1e-4,
    help = """learning rate""")

args.add_argument('--factor',
    type = float,
    default = .0005,
    help = """factor of regularization""")#正規化係數

args.add_argument('--batch_size',
    type = int,
    default = 8,
    help = """batch size for training""")#批次大小

args.add_argument('--weight_init',
    type = float,
    default = .1,
    help = """weight init for FC layers""")#FC層的權重

args.add_argument('--threshold',
    type = float,
    default = 48.0,
    help = """the threshold pass which is identified as hit""")#判斷hit的閾值