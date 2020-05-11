import torch
import spacy
from data_preprocessing import TEXT
from model import RNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_sentiment(model, sentence):
    model.eval()

    nlp = spacy.load('en')
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))

    return prediction.item()


if __name__ == '__main__':
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    model.load_state_dict(torch.load('tut1-model.pt'))

    print(predict_sentiment(model, 'This film is great'))
