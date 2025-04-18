import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim

from stanza.models.common import utils
from stanza.models.common.trainer import Trainer as BaseTrainer
from stanza.models.tokenization.utils import create_dictionary

from .model import Tokenizer
from .vocab import Vocab

logger = logging.getLogger('stanza')

class Trainer(BaseTrainer):
    def __init__(self, args=None, vocab=None, lexicon=None, dictionary=None, model_file=None, device=None):
        if model_file is not None:
            # load everything from file
            self.load(model_file)
        else:
            # build model from scratch
            self.args = args
            self.vocab = vocab
            self.lexicon = list(lexicon) if lexicon is not None else None
            self.dictionary = dictionary
            self.model = Tokenizer(self.args, self.args['vocab_size'], self.args['emb_dim'], self.args['hidden_dim'], dropout=self.args['dropout'], feat_dropout=self.args['feat_dropout'])
        self.model = self.model.to(device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
        self.optimizer = utils.get_optimizer("adam", self.model, lr=self.args['lr0'], betas=(.9, .9), weight_decay=self.args['weight_decay'])
        self.feat_funcs = self.args.get('feat_funcs', None)
        self.lang = self.args['lang'] # language determines how token normalization is done

    def update(self, inputs):
        self.model.train()
        units, labels, features, _ = inputs

        device = next(self.model.parameters()).device
        units = units.to(device)
        labels = labels.to(device)
        features = features.to(device)

        pred = self.model(units, features)

        self.optimizer.zero_grad()
        classes = pred.size(2)
        loss = self.criterion(pred.view(-1, classes), labels.view(-1))

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        self.optimizer.step()

        return loss.item()

    def predict(self, inputs):
        self.model.eval()
        units, _, features, _ = inputs

        device = next(self.model.parameters()).device
        units = units.to(device)
        features = features.to(device)

        pred = self.model(units, features)

        return pred.data.cpu().numpy()

    def save(self, filename):
        params = {
            'model': self.model.state_dict() if self.model is not None else None,
            'vocab': self.vocab.state_dict(),
            # save and load lexicon as list instead of set so
            # we can use weights_only=True
            'lexicon': list(self.lexicon) if self.lexicon is not None else None,
            'config': self.args
        }
        try:
            torch.save(params, filename, _use_new_zipfile_serialization=False)
            logger.info("Model saved to {}".format(filename))
        except BaseException:
            logger.warning("Saving failed... continuing anyway.")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename, lambda storage, loc: storage, weights_only=True)
        except BaseException:
            logger.error("Cannot load model from {}".format(filename))
            raise
        self.args = checkpoint['config']
        if self.args.get('use_mwt', None) is None:
            # Default to True as many currently saved models
            # were built with mwt layers
            self.args['use_mwt'] = True
        self.model = Tokenizer(self.args, self.args['vocab_size'], self.args['emb_dim'], self.args['hidden_dim'], dropout=self.args['dropout'], feat_dropout=self.args['feat_dropout'])
        self.model.load_state_dict(checkpoint['model'])
        self.vocab = Vocab.load_state_dict(checkpoint['vocab'])
        self.lexicon = checkpoint['lexicon']

        if self.lexicon is not None:
            self.lexicon = set(self.lexicon)
            self.dictionary = create_dictionary(self.lexicon)
        else:
            self.dictionary = None
