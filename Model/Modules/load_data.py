import pandas as pd

def load_data():
    train_df = pd.read_csv('/home/workspace/Dataset/train.csv')
    test_df = pd.read_csv('/home/workspace/Dataset/test.csv')
    sample_submission = pd.read_csv('/home/workspace/Dataset/sample_submission.csv')
    train_img_path = '/home/workspace/Dataset/image/train'
    test_img_path = '/home/workspace/Dataset/image/test'
    return train_df, test_df, sample_submission, train_img_path, test_img_path