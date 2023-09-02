import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from tqdm import tqdm
import numpy as np
import math
import re
from rouge_score import rouge_scorer
import torch.nn.functional as F

# from transformers import RobertaTokenizer, RobertaForSequenceClassification

class KaleidoSys:
    def __init__(self, model_name='tsor13/kaleido-small', embed_model_name='sentence-transformers/all-mpnet-base-v2', device="cuda" if torch.cuda.is_available() else "cpu", use_tqdm=True):
        self.model_name = model_name
        self.embed_model_name = embed_model_name
        self.device = device
        # model
        self.model = None
        self.tokenizer = None
        # BERT model (for deduplication)
        self.embed_model = None
        self.embed_tokenizer = None
        # templates
        self.gen_template, self.relevant_template, self.valence_template, self.explanation_template = None, None, None, None

        self.get_default_batch_sizes()

        self.use_tqdm = use_tqdm

    def tqdm(self, *args, **kwargs):
        if self.use_tqdm:
            return tqdm(*args, **kwargs)
        else:
            return args[0]

    def load_model(self):
        '''
        Load Kaleido model and initialize templates
        '''
        print('Loading model...')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        if torch.cuda.device_count() > 1:
            self.model.parallelize()
            # self device is model device
            self.device = self.model.device
        else:
            self.model = self.model.to(self.device)

        # to device
        self.device = self.model.device

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # get templates
        def get_template(task):
            text_template = self.model.config.task_specific_params[task]['template']
            task_args = self.model.config.task_specific_params[task]['args']
            # make a lambda function that takes in len(task_args) arguments and replaces them in the template
            def replace(text, *args):
                for arg, replacement in zip(task_args, args):
                    # if arg is VRD, make sure it's value, right, or duty
                    if arg == 'VRD':
                        replacement = self.clean_vrd(replacement)
                    text = text.replace(arg, replacement)
                return text
            return lambda *args: replace(text_template, *args)

        self.gen_template = get_template('generate')
        self.relevant_template = get_template('relevance')
        self.valence_template = get_template('valence')
        self.explanation_template = get_template('explanation')
            
        # TODO - update this
        tokens = ['Yes', 'No', 'Supports', 'Opposes', 'Either']
        self.token_ids = {token: self.tokenizer(token).input_ids[0] for token in tokens}
        self.relevant_ids = [self.token_ids['Yes'], self.token_ids['No']]
        self.index_to_relevant = {0: 'Yes', 1: 'No'}

        self.valence_ids = [self.token_ids['Supports'], self.token_ids['Opposes'], self.token_ids['Either']]
        self.index_to_valence = {0: 'Supports', 1: 'Opposes', 2: 'Either'}

    def get_default_batch_sizes(self):
        '''
        Function to get default batch sizes based on GPU memory
        '''
        self.bert_batch_size = 128
        if not torch.cuda.is_available():
            self.inference_batch_size = 8
            self.generation_batch_size = 1
            self.explanation_batch_size = 8
            return
        # get total memory
        # initialize total_memory
        total_memory = 0

        # iterate over all devices
        for i in range(torch.cuda.device_count()):
            total_memory += torch.cuda.get_device_properties(i).total_memory

        # if over 80GB (a100)
        if total_memory > 80_000_000_000:
            self.inference_batch_size = 512
            self.generation_batch_size = 2
            self.explanation_batch_size = 64
        # else, if over 50GB (a6000)
        elif total_memory > 50_000_000_000:
            # self.inference_batch_size = 128
            self.inference_batch_size = 256
            # self.generation_batch_size = 1
            self.generation_batch_size = 2
            self.explanation_batch_size = 64
        else:
            self.inference_batch_size = 32
            self.generation_batch_size = 1
            self.explanation_batch_size = 16

    def load_embedder(self):
        # TODO - add to params
        self.embed_tokenizer = AutoTokenizer.from_pretrained(self.embed_model_name)
        self.embed_model = AutoModel.from_pretrained(self.embed_model_name)
        # to device
        self.embed_model.to(self.device)

    def get_embeddings(self, texts):
        # if single, make into list
        is_single = False
        if isinstance(texts, str):
            texts = [texts]
            is_single = True

        if not self.embed_model or not self.embed_tokenizer:
            self.load_embedder()

        encoded_input = self.embed_tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)

        with torch.no_grad():
            model_output = self.embed_model(**encoded_input)

        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # if single, return single
        if is_single:
            sentence_embeddings = sentence_embeddings[0]

        return sentence_embeddings

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    
    def parse_gen(self, gen):
        vrd, text = gen.split(': ')[0], gen.split(': ')[1]
        return {'vrd': vrd, 'text': text}

    def parse_gens(self, gens):
        decoded = [self.parse_gen(gen) for gen in gens]
        return decoded

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def deduplicate(self, df, embed_threshold=None, relevance_threshold=None, ngram_threshold=None, dedup_across_vrd=False, rouge_type='rougeLsum', force_embeds=False):
        # remove by relevance
        df = self.remove_by_relevance(df, relevance_threshold=relevance_threshold)
        # dedup with ngrams
        df = self.deduplicate_rouge(df, rouge_type=rouge_type, threshold=ngram_threshold, dedup_across_vrd=dedup_across_vrd)
        # dedup with embeddings
        df = self.deduplicate_embed(df, threshold=embed_threshold, dedup_across_vrd=dedup_across_vrd, force_embeds=force_embeds)
        return df


    def remove_by_relevance(self, df, relevance_threshold=None):
        if relevance_threshold is None:
            relevance_threshold = self.model.config.system_params['relevance_threshold']
        if isinstance(relevance_threshold, dict):
            relevance_threshold_value = relevance_threshold.get('Value', self.model.config.system_params['relevance_threshold']['Value'])
            relevance_threshold_right = relevance_threshold.get('Right', self.model.config.system_params['relevance_threshold']['Right'])
            relevance_threshold_duty = relevance_threshold.get('Duty', self.model.config.system_params['relevance_threshold']['Duty'])
            if not set(relevance_threshold.keys()).issubset({'Value', 'Right', 'Duty'}):
                raise ValueError("Relevance threshold dictionary can only contain 'Value', 'Right', and 'Duty'")
        else:
            relevance_threshold_value = relevance_threshold_right = relevance_threshold_duty = relevance_threshold

        before_size = len(df)
        df = df[(df['vrd']=='Value') & (df['relevant']>=relevance_threshold_value) |
                (df['vrd']=='Right') & (df['relevant']>=relevance_threshold_right) |
                (df['vrd']=='Duty') & (df['relevant']>=relevance_threshold_duty)]
        after_size = len(df)
        print("Dropping {} rows because of relevance threshold".format(before_size-after_size))
        return df

    def deduplicate_rouge(self, df, rouge_type='rougeL', threshold=None, dedup_across_vrd=False):
        if threshold is None:
            threshold = self.model.config.system_params['ngram_threshold']
        # threshold to dict
        if isinstance(threshold, dict):
            # if f1 is present, rename to fmeasure
            if 'f1' in threshold:
                threshold['fmeasure'] = threshold.pop('f1')
            # make sure only has fmeasure, precision, recall
            if not set(threshold.keys()).issubset({'fmeasure', 'precision', 'recall'}):
                raise ValueError("Threshold dictionary can only contain 'fmeasure', 'precision', and 'recall'")
        else:
            threshold = {'fmeasure': threshold, 'precision': threshold, 'recall': threshold}

        # sort by relevance (for breaking ties)
        df = df.sort_values(by='relevant', ascending=False)
        df = df.reset_index(drop=True)
        
        scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
        
        def clean_example(example):
            # remove "Right to" or "Duty to" from beginning of example
            if example.startswith('Right to '):
                example = example.replace('Right to ', '', 1)
            elif example.startswith('Duty to '):
                example = example.replace('Duty to ', '', 1)
            return example

        rows_to_drop = []
        for i in range(1, len(df)):
            for j in range(i):
                if j not in rows_to_drop and (dedup_across_vrd or df.iloc[i]['vrd'] == df.iloc[j]['vrd']):
                    value_i = clean_example(df.iloc[i]['text'])
                    value_j = clean_example(df.iloc[j]['text'])

                    scores = scorer.score(value_i, value_j)[rouge_type]
                    if scores.fmeasure > threshold['fmeasure'] or scores.precision > threshold['precision'] or scores.recall > threshold['recall']:
                        rows_to_drop.append(i)
                        break
        print("Dropping {} rows due to ngram".format(len(rows_to_drop)))
        return df.drop(df.index[rows_to_drop])


    def deduplicate_embed(self, df, threshold=None, relevance_threshold=None, dedup_across_vrd=False, force_embeds=False):
        if threshold is None:
            threshold = self.model.config.system_params['embed_threshold']
        if isinstance(threshold, dict):
            threshold_value = threshold.get('Value', self.model.config.system_params['embed_threshold']['Value'])
            threshold_right = threshold.get('Right', self.model.config.system_params['embed_threshold']['Right'])
            threshold_duty = threshold.get('Duty', self.model.config.system_params['embed_threshold']['Duty'])
            # check if there are other thresholds - throw error if so
            if not set(threshold.keys()).issubset({'Value', 'Right', 'Duty'}):
                raise ValueError("Threshold dictionary can only contain 'Value', 'Right', and 'Duty'")
        else:
            threshold_value = threshold_right = threshold_duty = threshold

        # sort by relevance (for breaking ties)
        df = df.sort_values(by='relevant', ascending=False)
        df = df.reset_index(drop=True)
        # fill in embeddings if not present
        if 'embedding' not in df.columns or force_embeds:
            df['embedding'] = self.get_embeddings(df['text'].tolist()).tolist()

        rows_to_drop = []
        for i in range(1, len(df)):
            for j in range(i):
                # compare only to rows with same VRD if dedup_across_vrd is False
                if j not in rows_to_drop and (df.iloc[i]['vrd'] == df.iloc[j]['vrd'] or dedup_across_vrd):
                    sim = self.cosine_similarity(df.iloc[i]['embedding'], df.iloc[j]['embedding'])
                    if ((df.iloc[i]['vrd'] == 'Value' and sim > threshold_value) or
                    (df.iloc[i]['vrd'] == 'Right' and sim > threshold_right) or
                    (df.iloc[i]['vrd'] == 'Duty' and sim > threshold_duty)):
                        rows_to_drop.append(i)
                        break
        print("Dropping {} rows due to embedding similarity".format(len(rows_to_drop)))
        df = df.drop(columns=['embedding'])
        return df.drop(df.index[rows_to_drop])

    
    def clean_vrd(self, vrd):
        # check if value in vrd, no case
        if 'value' in vrd.lower():
            return 'Value'
        elif 'right' in vrd.lower():
            return 'Right'
        elif 'dut' in vrd.lower():
            return 'Duty'
        else:
            raise ValueError('Invalid VRD - {}. Must be "Value", "Right", or "Duty"'.format(vrd))

    
    def get_all_scores(self, actions, vrds, texts, explain=False, explanation_decoding_params={}):
        df = pd.DataFrame({'action': actions, 'vrd': vrds, 'text': texts})
        # clean vrds
        df['vrd'] = df['vrd'].apply(self.clean_vrd)
        # get relevance
        relevances = self.get_relevance(actions, vrds, texts)
        df['relevant'] = relevances[:, 0]
        # get valence
        valences = self.get_valence(actions, vrds, texts)
        df['supports'] = valences[:, 0]
        df['opposes'] = valences[:, 1]
        df['either'] = valences[:, 2]
        # explain
        if explain:
            df['explanation'] = self.get_explanation(actions, vrds, texts, explanation_decoding_params=explanation_decoding_params)
        elif explain:
            print('Warning: explanation not supported for this model')
        return df

    def get_output(self, action, n_gens=100, sample=False, explain=False, get_embeddings=False, explanation_decoding_params={}):
        gens = self.get_gens(action, n_gens=n_gens, sample=sample)
        # decode
        parsed = self.parse_gens(gens)
        # populate
        df = self.get_all_scores([action] * len(parsed), [d['vrd'] for d in parsed], [d['text'] for d in parsed], explain=explain, explanation_decoding_params=explanation_decoding_params)
        # sort by relevance
        df = df.sort_values(by='relevant', ascending=False)

        if get_embeddings:
            # get bert embeddings
            embeddings = self.get_embeddings(df['text'].tolist())
            df['embedding'] = embeddings.tolist()
            # make each numpy
            df['embedding'] = df['embedding'].apply(np.array)
        return df

    def get_valence_from_df(self, df):
        # get relevance
        relevance = df['relevant'].values
        # get supports, opposes, either
        valence = df[['supports', 'opposes', 'either']].values
        # multiply
        valence_probs = (relevance.reshape(-1, 1) * valence).sum(axis=0)
        # normalize
        valence_probs = valence_probs / valence_probs.sum()
        return valence_probs
    
    def get_str_from_df(self, df):
        # structure like this:
        # label is max between supports, opposes, either
        df['label'] = df[['supports', 'opposes', 'either']].values.argmax(axis=1)
        # map to support, oppose, either
        df['label'] = df['label'].map({0: 'supports', 1: 'opposes', 2: 'either'})
        
        s = ''
        # first, get all values
        # sort by relevance
        df = df.sort_values(by='relevant', ascending=False)
        for vrd, plural in [('Value', 'Values'), ('Right', 'Rights'), ('Duty', 'Duties')]:
            s += '{}:\n'.format(plural)
            # filter to where vrd == vrd
            df_vrd = df[df['vrd'] == vrd]
            # sort by relevance
            df_vrd = df_vrd.sort_values(by='relevant', ascending=False)
            # loop over values
            if len(df_vrd) == 0:
                s += '- N/A\n'
            else:
                for i, row in df_vrd.iterrows():
                    # get label
                    label = row['label']
                    # get text
                    text = row['text']
                    # check if explanation in df
                    if 'explanation' in row:
                        explanation = row['explanation']
                        # add to string
                        s += '- {}: {} [{}]\n'.format(text, explanation, label)
                    else:
                        # add to string
                        s += '- {} [{}]\n'.format(text, label)
            s += '\n'
        # strip
        s = s.strip()
        return s

    def get_candidates(self, action, explain=False, get_embeddings=True, explanation_decoding_params={}):
        pd.options.display.float_format = "{:,.2f}".format
        df = self.get_output(action, explain=explain, get_embeddings=get_embeddings, explanation_decoding_params=explanation_decoding_params)

        df = self.deduplicate(df, dedup_across_vrd=self.model.config.system_params['dedup_across_vrd'])

        print(action)
        print(self.get_str_from_df(df))
        # print(df)
        print(self.get_valence_from_df(df))
        print(df)
        return df

    def get_gens(self, actions, n_gens=100, sample=False, batch_size=None):
        if not self.model:
            self.load_model()
        # if one action, make it a list
        is_single = False
        if isinstance(actions, str):
            actions = [actions]
            is_single = True
        if batch_size is None:
            batch_size = self.generation_batch_size

        batch_gens = []
        n_batches = math.ceil(len(actions) / batch_size)
        for i in self.tqdm(range(n_batches), desc='Generation'):
            batch_actions = actions[i*batch_size:(i+1)*batch_size]
            encoded_batch = self.tokenizer.batch_encode_plus(
                [self.gen_template(action) for action in batch_actions],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128).to(self.device).input_ids
            with torch.no_grad():
                gens = self.model.generate(encoded_batch, num_beams=n_gens, num_return_sequences=n_gens, max_new_tokens=30)
            # decode
            gens = self.tokenizer.batch_decode(gens, skip_special_tokens=True)
            # add to list
            batch_gens.extend(gens)
        # reshape to (n_actions, n_gens)
        decoded = np.array(batch_gens).reshape((len(actions), n_gens))
        # strip
        decoded = np.array([np.array([x.strip() for x in gen]) for gen in decoded])
        # if single, return as string
        if is_single:
            decoded = decoded[0]
        return decoded
    
    def get_explanation(self, actions, vrds, texts, batch_size=None, explanation_decoding_params={}):
        # if max_length isn't present in explanation_decoding_params, set it to 128
        if 'max_length' not in explanation_decoding_params:
            explanation_decoding_params['max_length'] = 128
        print(explanation_decoding_params)
        # if one action, make it a list
        is_single = False
        if isinstance(actions, str):
            actions = [actions]
            vrds = [vrds]
            # clean vrds
            vrds = [self.clean_vrd(vrd) for vrd in vrds]
            texts = [texts]
            is_single = True
        if batch_size is None:
            batch_size = self.explanation_batch_size
        
        batch_exps = []
        n_batches = math.ceil(len(actions) / batch_size)
        for i in self.tqdm(range(n_batches), desc='Explanation'):
            batch_actions = actions[i*batch_size:(i+1)*batch_size]
            batch_vrds = vrds[i*batch_size:(i+1)*batch_size]
            batch_texts = texts[i*batch_size:(i+1)*batch_size]
            # get explanations
            encoded_batch = self.tokenizer.batch_encode_plus(
                [self.explanation_template(action, vrd, text) for action, vrd, text in zip(batch_actions, batch_vrds, batch_texts)],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128).to(self.device).input_ids
            with torch.no_grad():
                exps = self.model.generate(encoded_batch, **explanation_decoding_params)
            # decode
            exps = self.tokenizer.batch_decode(exps, skip_special_tokens=True)
            # add to list
            batch_exps.extend(exps)

        if is_single:
            batch_exps = batch_exps[0]
        return batch_exps


    def get_dummy(self, encoded_batch):
        # get dummy labels (0,0) * batch size
        dummy_labels = torch.tensor([[0, 0]] * encoded_batch.shape[0]).to(self.device)
        return dummy_labels

    def get_probs(self, inputs, batch_size=None):
        if not self.model:
            self.load_model()
        if batch_size is None:
            batch_size = self.inference_batch_size
        # max batch size
        n_batches = len(inputs) // batch_size
        logits = []
        # to numpy
        inputs = np.array(inputs)
        with torch.no_grad():
            # for i in tqdm(range(n_batches+1)):
            for i in self.tqdm(range(n_batches+1), desc='Inference'):
                # inds = list(range(i*batch_size, (i+1)*batch_size))
                inds = list(range(i*batch_size, min((i+1)*batch_size, len(inputs))))
                encoded_batch = self.tokenizer.batch_encode_plus(
                    inputs[inds].tolist(),
                    return_tensors='pt', padding=True, truncation=True, max_length=128,
                ).to(self.device).input_ids
                # batch_inputs = encoded_batch[i*batch_size:(i+1)*batch_size]
                # Run through model, get last logits
                logits_batch = self.model(input_ids=encoded_batch, labels=self.get_dummy(encoded_batch)).logits[:, -1, :].detach().cpu()
                logits.append(logits_batch)
        
        # concatenate logits
        logits = torch.cat(logits, dim=0)
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        return probs.cpu()

    def get_probs_template(self, actions, vrds, texts, template, token_ids, batch_size=None):
        # clean vrds
        vrds = [self.clean_vrd(vrd) for vrd in vrds]
        # templatize
        inputs = [template(a, v, val) for a, v, val in zip(actions, vrds, texts)]
        # pass through get_probs
        probs = self.get_probs(inputs, batch_size=batch_size)
        probs = probs[:, token_ids]
        # renormalize
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs.cpu()

    def get_relevance(self, actions, vrds, texts, batch_size=None):
        if not self.model:
            self.load_model()
        # check if str (if single instance, then batch)
        single = False
        if isinstance(actions, str):
            actions = [actions]
            vrds = [vrds]
            # clean vrds
            vrds = [self.clean_vrd(vrd) for vrd in vrds]
            texts = [texts]
            single = True
        # run through get_probs_template
        probs = self.get_probs_template(actions, vrds, texts, self.relevant_template, self.relevant_ids, batch_size=batch_size)
        if single:
            probs = probs[0]
        return probs

    def get_valence(self, actions, vrds, values, batch_size=None):
        if not self.model:
            self.load_model()
        # check if str (if single instance, then batch)
        single = False
        if isinstance(actions, str):
            actions = [actions]
            vrds = [vrds]
            # clean vrds
            vrds = [self.clean_vrd(vrd) for vrd in vrds]
            values = [values]
            single = True
        # run through get_probs_template
        probs = self.get_probs_template(actions, vrds, values, self.valence_template, self.valence_ids, batch_size=batch_size)
        if single:
            probs = probs[0]
        return probs

if __name__ == '__main__':
    system = KaleidoSys(model_name='tsor13/kaleido-small')

    while True:
        action = input('Action: ')
        system.get_candidates(action, explain=False)
