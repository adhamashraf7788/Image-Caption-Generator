from src.inference.predict import Predictor

predictor = Predictor(
    checkpoint_path='models/base_resnet_lstm/best_model.pt',
    vocab_path='data/processed/vocab.json',
    device='cuda',
)

image_path = 'data/raw/images/111766423_4522d36e56.jpg'
print('Greedy:', predictor.predict(image_path, decoding='greedy'))
print('Beam-3:', predictor.predict(image_path, decoding='beam'))
