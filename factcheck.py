from __future__ import annotations

import re
import gc
from typing import List, Tuple, Dict, Iterable, Set

import numpy as np
import torch


class TextProcessingUtils:

    TOKEN_REGEX = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

    STOPWORD_SET = {
        "a","an","the","and","or","but","if","then","than","that","this","those","these","to","of","in","on","for","at",
        "by","with","as","is","are","was","were","be","been","being","from","it","its","into","about","over","under",
        "he","she","they","them","his","her","their","we","you","i","me","my","your","our","ours","yours","not","no"
    }

    NUMBER_REGEX = re.compile(r"\b\d{1,4}\b")

    @staticmethod
    def tokenize(text: str, *, lowercase: bool=True, remove_stop: bool=True) -> List[str]:
        if not text:
            return []
        if lowercase:
            text = text.lower()
        toks = TextProcessingUtils.TOKEN_REGEX.findall(text)
        if remove_stop:
            toks = [t for t in toks if t not in TextProcessingUtils.STOPWORD_SET]
        return toks

    @staticmethod
    def sent_split(text: str) -> List[str]:
        if not text:
            return []
        t = re.sub(r"</?s>", " ", text)
        t = re.sub(r"\s+", " ", t).strip()
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", t)
        if len(pieces) == 1:

            pieces = re.split(r"[\n]+|(?<=[.!?])", t)
            pieces = [p.strip() for p in pieces if p.strip()]
        else:
            pieces = [p.strip() for p in pieces if p.strip()]
        return pieces

    @staticmethod
    def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
        A = set(a); B = set(b)
        if not A and not B:
            return 0.0
        inter = len(A & B); uni = len(A | B)
        return inter / uni if uni else 0.0

    @staticmethod
    def recall(fact_tokens: Iterable[str], text_tokens: Iterable[str]) -> float:
        F = set(fact_tokens); T = set(text_tokens)
        if not F:
            return 0.0
        return len(F & T) / len(F)

    @staticmethod
    def numbers(text: str) -> set:
        return set(TextProcessingUtils.NUMBER_REGEX.findall(text or ""))

    @staticmethod
    def bigrams(tokens: List[str]) -> set:
        """Generates a set of bigrams from a list of tokens."""
        return set(zip(tokens, tokens[1:])) if len(tokens) >= 2 else set()



class FactExample(object):
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self) -> str:
        return f"FactExample(fact={self.fact!r}, passages={len(self.passages)} passages, label={self.label!r})"

    def __str__(self) -> str:
        return self.__repr__()


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda: bool=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str) -> Dict[str, float]:
        with torch.no_grad():
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512,
            )
            if self.cuda:
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

        del inputs, outputs, logits
        gc.collect()

        id2label = getattr(self.model.config, 'id2label', {0:'entailment',1:'neutral',2:'contradiction'})
        out = { 'entailment': 0.0, 'neutral': 0.0, 'contradiction': 0.0 }
        for idx, p in enumerate(probs):
            lab = id2label.get(idx, str(idx)).lower()
            if 'entail' in lab:
                lab = 'entailment'
            elif 'contrad' in lab:
                lab = 'contradiction'
            elif 'neutral' in lab:
                lab = 'neutral'
            else:
                lab = 'neutral'
            out[lab] = float(p)
        return out


class FactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def __init__(self,
                 threshold: float = 0.50,
                 jaccard_gate: float = 0.18,
                 require_bigram_or_jaccard: bool = True,
                 require_number_consistency: bool = True):
        self.threshold = threshold
        self.jaccard_gate = jaccard_gate
        self.require_bigram_or_jaccard = require_bigram_or_jaccard
        self.require_number_consistency = require_number_consistency

    def _num_consistent(self, fact: str, sent: str) -> bool:
        if not self.require_number_consistency:
            return True
        fnums = TextProcessingUtils.numbers(fact)
        if not fnums:
            return True
        return len(fnums & TextProcessingUtils.numbers(sent)) > 0

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_toks = TextProcessingUtils.tokenize(fact)
        if not fact_toks:
            return "NS"

        fact_bi = TextProcessingUtils.bigrams(fact_toks)
        best_recall = 0.0

        for p in passages:
            text = p.get("text", "") or ""
            for sent in TextProcessingUtils.sent_split(text):
                sent_toks = TextProcessingUtils.tokenize(sent)
                if not sent_toks:
                    continue

                rec = TextProcessingUtils.recall(fact_toks, sent_toks)
                if rec > best_recall:
                    best_recall = rec

                if rec < self.threshold:
                    continue

                gate_ok = True
                if self.require_bigram_or_jaccard:
                    shared_bi = len(fact_bi & TextProcessingUtils.bigrams(sent_toks))
                    jac = TextProcessingUtils.jaccard(fact_toks, sent_toks)
                    gate_ok = (shared_bi > 0) or (jac >= self.jaccard_gate)

                if gate_ok and self._num_consistent(fact, sent):
                    return "S"

        return "S" if best_recall >= (self.threshold + 0.05) else "NS"


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model: EntailmentModel,
                 entail_threshold: float = 0.44,
                 prune_recall: float = 0.0,
                 use_pair_windows: bool = True,
                 max_pair_chars: int = 360):
        self.ent_model = ent_model
        self.entail_threshold = entail_threshold
        self.prune_recall = prune_recall
        self.use_pair_windows = use_pair_windows
        self.max_pair_chars = max_pair_chars

    def _candidates(self, text: str) -> List[str]:
        sents = TextProcessingUtils.sent_split(text)
        cands = list(sents)
        if self.use_pair_windows and len(sents) >= 2:
            for i in range(len(sents) - 1):
                pair = (sents[i] + " " + sents[i+1]).strip()
                if len(pair) <= self.max_pair_chars:
                    cands.append(pair)
        return cands

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_toks = TextProcessingUtils.tokenize(fact)
        if not fact_toks:
            return "NS"

        best_entail_p = 0.0

        for p in passages:
            text = p.get("text", "") or ""
            for cand in self._candidates(text):

                if TextProcessingUtils.recall(fact_toks, TextProcessingUtils.tokenize(cand)) < self.prune_recall:
                    continue

                probs = self.ent_model.check_entailment(premise=cand, hypothesis=fact)
                pe = probs["entailment"]
                if pe > best_entail_p:
                    best_entail_p = pe
                    if best_entail_p >= 0.90:
                        return "S"

        return "S" if best_entail_p >= self.entail_threshold else "NS"


try:
    import spacy
    _HAVE_SPACY = True
except Exception:
    _HAVE_SPACY = False


class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        if _HAVE_SPACY:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except Exception:
                self.nlp = spacy.blank("en")
        else:
            self.nlp = None

    def predict(self, fact: str, passages: List[dict]) -> str:
        if self.nlp is None or not hasattr(self, "get_dependencies"):
            return WordRecallThresholdFactChecker(threshold=0.52).predict(fact, passages)

        fact_rels = self.get_dependencies(fact)
        if not fact_rels:
            return "NS"

        best_recall = 0.0
        for p in passages:
            text = p.get("text", "") or ""
            for sent in TextProcessingUtils.sent_split(text):
                rels = self.get_dependencies(sent)
                if not rels:
                    continue
                inter = len(fact_rels & rels)

                if len(fact_rels) == 0:
                    continue
                rec = inter / len(fact_rels)
                if rec > best_recall:
                    best_recall = rec
                    if best_recall >= self.threshold:
                        return "S"
        return "S" if best_recall >= self.threshold else "NS"

    def get_dependencies(self, sent: str):
        if self.nlp is None:
            return set()
        processed_sent = self.nlp(sent)
        relations = set()
        ignore_dep = {'punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark', 'cc'}
        for token in processed_sent:
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head.lower(), token.dep_.lower(), dependent.lower())
            relations.add(relation)
        return relations