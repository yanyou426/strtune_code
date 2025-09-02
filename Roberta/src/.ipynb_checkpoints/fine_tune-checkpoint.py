from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import pickle,math
from numpy import dot
from numpy.linalg import norm
from transformers import RobertaTokenizerFast
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import torch


batch_size = 8
num_epoch = 32 # 32

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def load_data():
    with open("./data/pair_update/test.pkl", "rb") as f:
        test = pickle.load(f)
    with open("./data/pair_update/train.pkl", "rb") as f:
        train = pickle.load(f)
    test_samples = []
    for pair in test:
        label = 1.0 if pair[2] else 0.0
        inp_exp = InputExample(texts=[pair[0][:510], pair[1][:510]], label=label)
        test_samples.append(inp_exp)
    train_samples = []
    for pair in train:
        label = 1.0 if pair[2] else 0.0
        inp_exp = InputExample(texts=[pair[0][:510], pair[1][:510]], label=label)
        train_samples.append(inp_exp)
    return train_samples, test_samples


def finetune():
    
#     wd_model = models.Transformer("./BERT/checkpoint-1600000")
#     pooling_model = models.Pooling(wd_model.get_word_embedding_dimension(),
#                                pooling_mode_mean_tokens=True,
#                                pooling_mode_cls_token=False,
#                                pooling_mode_max_tokens=False)
#     model = SentenceTransformer(modules=[wd_model, pooling_model])
    
    
    # model = SentenceTransformer("./BERT/checkpoint-1600000") # same as the initial version, model consisting of Transformer and pooling 
    # print(model)
    
    
    train_loss = losses.CosineSimilarityLoss(model=model)
    train_samples, test_samples = load_data()
    # train_samples += load_data2()
    print(f"Num of train: {len(train_samples)}")
    train_data_loader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-dev')
    warmup_steps = math.ceil(len(train_data_loader) * num_epoch  * 0.1) #10% of train data for warm-up
    print(f"Warmup steps: {warmup_steps}")
    model.fit(train_objectives=[(train_data_loader, train_loss)],
          evaluator=evaluator,
          epochs=num_epoch,
          evaluation_steps=1,
          warmup_steps=warmup_steps,
          output_path="./BERT-ft")

def run():
    model = SentenceTransformer("./BERT-ft")  

    sentences1 = 'ldx ds (var+(constant*eax))'
    sentence_embedding = model.encode(sentences1)
    print(sentence_embedding.shape) # 1 * 768

    
def model_encode(sentences):



    #Sentences we want sentence embeddings for
    

    #Load AutoModel from huggingface model repository
    tokenizer = RobertaTokenizerFast(tokenizer_file='BERT/vocab.json')
    model = AutoModel.from_pretrained("./BERT/checkpoint-1600000")

    #Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=512, return_tensors='pt')
    # print(encoded_input)

    #Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    #Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings
    



if __name__ == "__main__":
    sentences = ['ldx ds (var+(constant*eax))']
    print(model_encode(sentences).shape)
    
    # finetune()
    # run()
