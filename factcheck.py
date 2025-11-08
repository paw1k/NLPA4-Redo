
import torch
from typing import List
import numpy as np
import spacy
import gc
import re
from torch.nn.functional import softmax

# A simple list of stopwords
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', 'should', 'now'
}

class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True, max_length=512)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        # Store the result on CPU before deleting CUDA tensors
        result_logits = logits.cpu()

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        if self.cuda:
            torch.cuda.empty_cache() # Explicitly empty cache for GPU
        gc.collect()

        return result_logits

class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):

    def _preprocess(self, text: str) -> set:
        """
        Lowercase, tokenize, remove stopwords, and return a set of tokens.
        """
        # Find all word sequences
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Filter out stopwords
        return {token for token in tokens if token not in STOPWORDS}

    def predict(self, fact: str, passages: List[dict]) -> str:
        # Per handout Part 1, we tune a threshold on the dev set.
        # A threshold of 0.75 is a reasonable starting point.
        threshold = 0.75

        fact_tokens = self._preprocess(self.fact)

        # If the fact has no processable tokens, it can't be supported.
        if not fact_tokens:
            return "NS"

        # Combine all passage text into one big string and preprocess
        all_passage_text = " ".join([passage['text'] for passage in passages])
        passage_tokens = self._preprocess(all_passage_text)

        # Calculate the intersection (overlapping tokens)
        overlap = fact_tokens.intersection(passage_tokens)

        # Calculate recall: (overlapping tokens) / (total tokens in fact)
        recall = len(overlap) / len(fact_tokens)

        # Classify based on the recall threshold
        if recall >= threshold:
            return "S"
        else:
            return "NS"

class EntailmentFactChecker(FactChecker):

    def __init__(self, ent_model):
        self.ent_model = ent_model

        # Load spacy model for sentence splitting
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except IOError:
            print("Error: Spacy model 'en_core_web_sm' not found.")
            print("Please run: python -m spacy download en_core_web_sm")
            self.nlp = None # Will cause failure later, but initialization won't crash

        # Thresholds (tunable, based on handout)
        # Low recall threshold for pruning (per handout optimization)
        self.pruning_recall_threshold = 0.1
        # Classification threshold: S if (entail_prob - contra_prob) >= threshold
        self.classification_threshold = 0.0

    def _preprocess(self, text: str) -> set:
        """
        Helper for pruning: Lowercase, tokenize, remove stopwords.
        """
        tokens = re.findall(r'\b\w+\b', text.lower())
        return {token for token in tokens if token not in STOPWORDS}

    def _get_sentences(self, text: str) -> List[str]:
        """
        Cleans text and splits into sentences using Spacy.
        """
        # Clean Wikipedia noise (per handout example)
        cleaned_text = text.replace("<s>", " ").replace("</s>", " ").strip()

        if not self.nlp or not cleaned_text:
            return []

        doc = self.nlp(cleaned_text)
        return [str(sent).strip() for sent in doc.sents if str(sent).strip()]

    def predict(self, fact: str, passages: List[dict]) -> str:

        # --- 1. Pruning Step (Optimization per Part 2) ---
        fact_tokens = self._preprocess(fact)
        if not fact_tokens:
            return "NS" # Empty fact

        all_passage_text = " ".join([p['text'] for p in passages])
        passage_tokens = self._preprocess(all_passage_text)

        if not passage_tokens:
            return "NS" # Empty passages

        # Calculate word recall for pruning
        recall = len(fact_tokens.intersection(passage_tokens)) / len(fact_tokens)

        # If overlap is too low, prune and return "NS"
        if recall < self.pruning_recall_threshold:
            return "NS"

        # --- 2. Entailment Step (if not pruned) ---
        # We are looking for the "max" score (per handout)
        max_score = -float('inf')

        for passage in passages:
            # Split passage into sentences
            sentences = self._get_sentences(passage['text'])

            for sent in sentences:
                premise = sent
                hypothesis = fact

                # Get logits from the entailment model
                logits = self.ent_model.check_entailment(premise, hypothesis)

                # Apply softmax to get probabilities
                # logits is [1, 3], so probs is [1, 3]. [0] gets the [entail, neutral, contra] tensor
                probs = softmax(logits, dim=-1)[0]

                # Label mapping: 0:"entailment", 1:"neutral", 2:"contradiction"
                entail_prob = probs[0].item()
                contra_prob = probs[2].item()

                # Use (entailment - contradiction) as the score
                # This is a common way to map 3-class NLI to a 2-class decision
                score = entail_prob - contra_prob

                if score > max_score:
                    max_score = score

        # Classify based on the highest score found across all sentences
        if max_score >= self.classification_threshold:
            return "S"
        else:
            return "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
