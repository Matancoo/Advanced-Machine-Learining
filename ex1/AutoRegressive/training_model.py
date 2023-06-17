"""
Trains a character-level language model.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
from mingpt.bpe import BPETokenizer

DATASET_PATH = "/Users/matancohen/Desktop/AML/ex1/AutoRegressive/alice_in_wonderland.txt"
# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'#TODO:change here

    # data
    C.data = TextDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    C.trainer.losses = []
    C.trainer.max_iters = 2000

    return C

# -----------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    Emits batches of characters
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128 # chunk size
        return C

    def __init__(self, config, data):
        self.config = config
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print("Stuff about data")
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        self.data = data.squeeze(0)

    def get_block_size(self):
        return self.config.block_size


    def __len__(self):
        return len(self.data) - self.config.block_size #TODO: check if this is right

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        # sliding this window one token at a time across the dataset
        chunk = self.data[idx:idx + self.config.block_size + 1]

        # return as tensors
        x = chunk[:-1].clone().detach().long() # tokens used to make prediction
        y = chunk[1:].clone().detach().long()  # token to predict next (shifted chunk by 1)
        return x, y

def invert_model(model, tokenizer, target_sentence,config,training_steps,lr):
    model.eval()
    device = next(model.parameters()).device

    tok_target_sentence = tokenizer(target_sentence)
    tok_target_sentence = tok_target_sentence.to(device)
    batch, block_size, n_embd = 1, len(tok_target_sentence[0]), config.model.n_embd

    input_vector = torch.randn((batch, block_size, n_embd), requires_grad=True,device=device)
    optimizer = torch.optim.Adam([input_vector], lr=lr)
    # inversion_process
    for _ in range(training_steps):
        logits  = model.token_embb_bypass_forward(input_vec=input_vector)

        # compute the loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tok_target_sentence.view(-1), ignore_index=-1)

        # backpropagate the gradients
        loss.backward()

        # update the input vector
        optimizer.step()

        # zero the gradients
        optimizer.zero_grad()
    return input_vector

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    text = open(DATASET_PATH, 'r').read()
    tokenizer = BPETokenizer()
    tokenized_text = tokenizer(text)
    train_dataset = TextDataset(config.data, tokenized_text)

    # construct the model
    config.model.vocab_size = len(tokenizer.encoder.encoder) #TODO: check if this the right size
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)
    #TODO: change the vocab size to be only what exist in text for EFFICIENCY
    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # iteration callback
    def batch_end_callback(trainer):

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        # if trainer.iter_num % 500 == 0: #TODO: do i need any evaluation ? or just append loss form model?
        #     # evaluate both the train and test score
        #     model.eval()
        #     with torch.no_grad():
        #         # sample from the model...
                # context = "O God, O God!"
                # x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
                # y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
                # completion = ''.join([train_dataset.itos[int(i)] for i in y])
                # print(completion)
            # save the latest model
            # print("saving model")
            # ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            # torch.save(model.state_dict(), ckpt_path)
            # # revert model to training mode
            # model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    # trainer.run()
    # save model
    save_path = os.path.join(config.system.work_dir, "model.pt")
    torch.save(model.state_dict(), save_path)

    #
    # target_sentence = "I am a little squirrel holding a walnut"
    # output_vector = invert_model(model, tokenizer, target_sentence, config, training_steps=1000, lr=0.1)
    # # check if the output vector is the same as the target sentence
    # output_logits = model.token_embb_bypass_forward(output_vector)
    # output_token_indices = torch.argmax(output_logits,dim=-1)
    # output_sentence = tokenizer.decode(output_token_indices[0])
    # print(output_sentence)

    # Q3
    # ar_generation(model, tokenizer, config, target_sentence, 1000, 0.1)
    model.eval()
    sentence = "I am a squirrel holding a walnut and I love it"
    tok_sentence = tokenizer(sentence)
    tok_sentence = tok_sentence#.to(device)
    #TODO: check that step size is good/
    #TODO: use the debug_tokenize function to see data
    #TODO: see chatGPT explainiation
    #TODO finish this fucking exercises.
    #TODO: name files correctly and create two main files as specidied
    #TODO: pass all questions answers to separate files
    output_idx = model.generate(tok_sentence, 1, temperature=1.0, do_sample=True, top_k=10)[0]
