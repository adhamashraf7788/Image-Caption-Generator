from src.inference.predict import Predictor
import pandas as pd

predictor = Predictor(
    checkpoint_path='models/base_resnet_lstm/best_model.pt',
    vocab_path='data/processed/vocab.json',
    device='cuda',
)

test_df = pd.read_csv('data/processed/test.csv')
sample_image = test_df.iloc[0]['image']
sample_refs = test_df[test_df['image'] == sample_image]['caption'].tolist()

caption = predictor.predict(f'data/raw/images/{sample_image}')
print('Image:', sample_image)
print('Generated repr:', repr(caption))
print('References:')
for r in sample_refs:
    print(' -', r)