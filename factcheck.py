from __future__ import annotations

import re
import gc
from typing import List, Tuple, Dict, Iterable, Set

import numpy as np
import torch

class TextProcessingUtils:
    """
    A utility class containing static methods for text processing,
    such as tokenization, sentence splitting, and set-based metrics.
    This replaces the global helper functions from the original code.
    """
    # Regex for finding words, including simple contractions
    TOKEN_REGEX = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

    # Set of common stopwords for filtering
    STOPWORD_SET = {
        "a","an","the","and","or","but","if","then","than","that","this","those","these","to","of","in","on","for","at",
        "by","with","as","is","are","was","were","be","been","being","from","it","its","into","about","over","under",
        "he","she","they","them","his","her","their","we","you","i","me","my","your","our","ours","yours","not","no"
    }

    @staticmethod
    def tokenize_text(text: str, *, lowercase: bool = True, remove_stop: bool = True) -> List[str]:
        """Tokenizes text, with options for lowercasing and stopword removal."""
        if not text:
            return []
        if lowercase:
            text = text.lower()
        tokens = TextProcessingUtils.TOKEN_REGEX.findall(text)
        if remove_stop:
            tokens = [tok for tok in tokens if tok not in TextProcessingUtils.STOPWORD_SET]
        return tokens

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """A fast, regex-based sentence splitter."""
        if not text:
            return []
        # Clean up Wikipedia tags and extra whitespace
        cleaned_text = re.sub(r"</?s>", " ", text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        # Primary split strategy
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned_text)

        # Fallback for single-block text (e.g., newline-separated)
        if len(sentences) == 1:
            sentences = re.split(r"[\n]+|(?<=[.!?])", cleaned_text)

        # Final cleanup
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def calculate_jaccard(set_a: Iterable[str], set_b: Iterable[str]) -> float:
        """Calculates Jaccard similarity between two iterables."""
        A = set(set_a); B = set(set_b)
        if not A and not B:
            return 0.0
        intersection = len(A & B)
        union = len(A | B)
        return intersection / union if union else 0.0

    @staticmethod
    def calculate_recall(fact_tokens: Iterable[str], text_tokens: Iterable[str]) -> float:
        """Calculates recall of fact_tokens present in text_tokens."""
        FactSet = set(fact_tokens)
        TextSet = set(text_tokens)
        if not FactSet:
            return 0.0
        return len(FactSet & TextSet) / len(FactSet)

    @staticmethod
    def extract_numbers(text: str) -> Set[str]:
        """Extracts 1-4 digit numbers from text."""
        return set(re.findall(r"\b\d{1,4}\b", text or ""))

    @staticmethod
    def generate_bigrams(tokens: List[str]) -> Set[Tuple[str, str]]:
        """Generates a set of bigrams from a list of tokens."""
        return set(zip(tokens, tokens[1:])) if len(tokens) >= 2 else set()


class EvidenceItem(object):
    """
    Data class representing a single claim to be verified against
    a list of source documents.
    """
    def __init__(self, claim: str, sources: List[dict], verdict: str):
        self.claim = claim
        self.sources = sources
        self.verdict = verdict # S, NS, or IR

    def __repr__(self) -> str:
        return f"EvidenceItem(claim={self.claim!r}, sources={len(self.sources)} sources, verdict={self.verdict!r})"

    def __str__(self) -> str:
        return self.__repr__()


class NLIWrapper(object):
    """
    A wrapper for the Hugging Face entailment model, handling
    tokenization, inference, and output normalization.
    """
    def __init__(self, model, tokenizer, use_gpu: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.use_gpu = use_gpu

    def query_nli_model(self, context: str, statement: str) -> Dict[str, float]:
        """
        Runs a premise (context) and hypothesis (statement) through
        the NLI model and returns a dictionary of probabilities.
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                context,
                statement,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=512,
            )
            if self.use_gpu:
                inputs = {key: val.to('cuda') for key, val in inputs.items()}

            outputs = self.model(**inputs)
            logits = outputs.logits

            # Get probabilities and move to CPU
            probabilities_tensor = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

        # Explicitly delete tensors and run garbage collection
        del inputs, outputs, logits
        if self.use_gpu:
            torch.cuda.empty_cache()
        gc.collect()

        # Normalize output labels
        id_to_label_map = getattr(self.model.config, 'id2label', {0:'entailment', 1:'neutral', 2:'contradiction'})
        probabilities = {'entailment': 0.0, 'neutral': 0.0, 'contradiction': 0.0}

        for index, probability in enumerate(probabilities_tensor):
            label = id_to_label_map.get(index, str(index)).lower()
            if 'entail' in label:
                norm_label = 'entailment'
            elif 'contrad' in label:
                norm_label = 'contradiction'
            else:
                norm_label = 'neutral'
            probabilities[norm_label] = float(probability)

        return probabilities


class ClaimVerifier(object):
    """Abstract base class for all fact-checking/verification models."""
    def predict(self, claim: str, sources: List[dict]) -> str:
        """
        Predicts whether a claim is Supported ("S") or Not Supported ("NS")
        based on the provided source passages.
        """
        raise NotImplementedError("Subclass must implement the 'predict' method.")


class RandomVerifier(ClaimVerifier):
    """Baseline model: Predicts "S" or "NS" at random."""
    def predict(self, claim: str, sources: List[dict]) -> str:
        return np.random.choice(["S", "NS"])


class SupportVerifier(ClaimVerifier):
    """Baseline model: Always predicts "S" (Supported)."""
    def predict(self, claim: str, sources: List[dict]) -> str:
        return "S"


class HeuristicVerifier(ClaimVerifier):
    """
    A verifier based on word recall and other heuristics like
    Jaccard similarity, bigram overlap, and number consistency.
    """
    def __init__(self,
                 recall_min: float = 0.50,
                 jaccard_min: float = 0.18,
                 require_gates: bool = True,
                 require_numbers: bool = True):
        self.recall_min = recall_min
        self.jaccard_min = jaccard_min
        self.require_gates = require_gates
        self.require_numbers = require_numbers

    def _check_numeric_consistency(self, claim_text: str, sentence_text: str) -> bool:
        """Checks if at least one number from the claim appears in the sentence."""
        if not self.require_numbers:
            return True
        claim_numbers = TextProcessingUtils.extract_numbers(claim_text)
        if not claim_numbers:
            return True # No numbers to check

        sentence_numbers = TextProcessingUtils.extract_numbers(sentence_text)
        return len(claim_numbers & sentence_numbers) > 0

    def predict(self, claim: str, sources: List[dict]) -> str:
        claim_toks = TextProcessingUtils.tokenize_text(claim)
        if not claim_toks:
            return "NS"

        claim_bigrams = TextProcessingUtils.generate_bigrams(claim_toks)
        top_recall_score = 0.0

        for source in sources:
            text = source.get("text", "") or ""
            for sentence in TextProcessingUtils.split_into_sentences(text):
                sentence_toks = TextProcessingUtils.tokenize_text(sentence)
                if not sentence_toks:
                    continue

                recall_score = TextProcessingUtils.calculate_recall(claim_toks, sentence_toks)

                if recall_score > top_recall_score:
                    top_recall_score = recall_score

                if recall_score < self.recall_min:
                    continue

                # Check additional heuristic gates if required
                passes_gates = True
                if self.require_gates:
                    common_bigrams = len(claim_bigrams & TextProcessingUtils.generate_bigrams(sentence_toks))
                    jaccard_score = TextProcessingUtils.calculate_jaccard(claim_toks, sentence_toks)
                    passes_gates = (common_bigrams > 0) or (jaccard_score >= self.jaccard_min)

                if passes_gates and self._check_numeric_consistency(claim, sentence):
                    return "S" # Found a strongly matching sentence

        # Fallback check: If the best recall score was *just* under the wire
        # or missed the gates, we give it a slight buffer.
        return "S" if top_recall_score >= (self.recall_min + 0.05) else "NS"


class NliVerifier(ClaimVerifier):
    """
    A verifier using a pre-trained NLI (entailment) model.
    Includes pruning and sentence-windowing optimizations.
    """
    def __init__(self, nli_model: NLIWrapper,
                 support_threshold: float = 0.44,
                 pruning_recall_cutoff: float = 0.04,
                 use_sentence_pairs: bool = True,
                 max_pair_length_chars: int = 360):
        self.nli_model = nli_model
        self.support_threshold = support_threshold
        self.pruning_recall_cutoff = pruning_recall_cutoff
        self.use_sentence_pairs = use_sentence_pairs
        self.max_pair_length_chars = max_pair_length_chars

    def _generate_text_chunks(self, text: str) -> List[str]:
        """Creates a list of sentences and (optionally) sentence-pairs."""
        sentences = TextProcessingUtils.split_into_sentences(text)
        chunks = list(sentences)

        if self.use_sentence_pairs and len(sentences) >= 2:
            for i in range(len(sentences) - 1):
                sentence_pair = (sentences[i] + " " + sentences[i+1]).strip()
                if len(sentence_pair) <= self.max_pair_length_chars:
                    chunks.append(sentence_pair)
        return chunks

    def predict(self, claim: str, sources: List[dict]) -> str:
        claim_toks = TextProcessingUtils.tokenize_text(claim)
        if not claim_toks:
            return "NS"

        max_entail_prob = 0.0

        for source in sources:
            text = source.get("text", "") or ""
            for chunk in self._generate_text_chunks(text):

                # Pruning step: check recall before running the NLI model
                chunk_toks = TextProcessingUtils.tokenize_text(chunk)
                if TextProcessingUtils.calculate_recall(claim_toks, chunk_toks) < self.pruning_recall_cutoff:
                    continue

                # Run the expensive NLI model
                probabilities = self.nli_model.query_nli_model(context=chunk, statement=claim)
                entail_prob = probabilities["entailment"]

                if entail_prob > max_entail_prob:
                    max_entail_prob = entail_prob

                    # Early exit optimization: if confidence is very high
                    if max_entail_prob >= 0.90:
                        return "S"

        # Final decision based on the best score found
        return "S" if max_entail_prob >= self.support_threshold else "NS"

# --- Optional Dependency-Based Verifier ---

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False


class DependencyVerifier(ClaimVerifier):
    """
    (Optional) A verifier that computes recall over dependency-parse
    relations instead of simple words.
    """
    def __init__(self, min_recall: float = 0.5):
        self.min_recall = min_recall
        self.nlp = None
        if _SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except Exception:
                # Fallback to a blank model if 'en_core_web_sm' isn't downloaded
                self.nlp = spacy.blank("en")

    def predict(self, claim: str, sources: List[dict]) -> str:
        if self.nlp is None or not hasattr(self, "extract_relations"):
            # Fallback to heuristic verifier if spaCy fails or method is missing
            return HeuristicVerifier(recall_min=0.52).predict(claim, sources)

        claim_relations = self.extract_relations(claim)
        if not claim_relations:
            return "NS"

        max_recall = 0.0
        for source in sources:
            text = source.get("text", "") or ""
            for sentence in TextProcessingUtils.split_into_sentences(text):
                sentence_relations = self.extract_relations(sentence)
                if not sentence_relations:
                    continue

                intersection_size = len(claim_relations & sentence_relations)
                recall_val = intersection_size / len(claim_relations)

                if recall_val > max_recall:
                    max_recall = recall_val
                    if max_recall >= self.min_recall:
                        return "S"

        return "S" if max_recall >= self.min_recall else "NS"

    def extract_relations(self, text: str) -> Set[Tuple[str, str, str]]:
        """Extracts a set of (head, dep_label, child) relations."""
        if self.nlp is None:
            return set()

        processed_doc = self.nlp(text)
        relations = set()

        # Define dependencies to ignore
        ignored_dependencies = {'punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark', 'cc'}

        for token in processed_doc:
            if token.is_punct or token.dep_ in ignored_dependencies:
                continue

            # Lemmatize verbs, use text otherwise
            head_text = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dep_text = token.lemma_ if token.pos_ == 'VERB' else token.text

            # Normalize to lowercase
            rel_tuple = (head_text.lower(), token.dep_.lower(), dep_text.lower())
            relations.add(rel_tuple)

        return relations