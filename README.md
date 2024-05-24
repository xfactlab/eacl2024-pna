# Capturing the Relationship Between Sentence Triplets for LLM and Human-Generated Texts to Enhance Sentence Embeddings

## Citation
We developed our implementation using the source code from [SimCSE](https://github.com/princeton-nlp/SimCSE) and [CLAIF](https://github.com/xiami2019/CLAIF).
```bibtex
@inproceedings{an-etal-2024-capturing,
    title = "Capturing the Relationship Between Sentence Triplets for {LLM} and Human-Generated Texts to Enhance Sentence Embeddings",
    author = "An, Na Min  and
      Waheed, Sania  and
      Thorne, James",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-eacl.43",
    pages = "624--638",
    abstract = "Deriving meaningful sentence embeddings is crucial in capturing the semantic relationship between texts. Recent advances in building sentence embedding models have centered on replacing traditional human-generated text datasets with those generated by LLMs. However, the properties of these widely used LLM-generated texts remain largely unexplored. Here, we evaluate the quality of the LLM-generated texts from four perspectives (Positive Text Repetition, Length Difference Penalty, Positive Score Compactness, and Negative Text Implausibility) and find that there exists an inherent difference between human and LLM-generated datasets. To further enhance sentence embeddings using both human and LLM-generated datasets, we propose a novel loss function that incorporates Positive-Negative sample Augmentation (PNA) within the contrastive learning objective. Our results demonstrate that PNA effectively mitigates the sentence anisotropy problem in Wikipedia corpus (-7{\%} compared to CLHAIF) and simultaneously improves the Spearman{'}s correlation in standard Semantic Textual Similarity (STS) tasks (+1.47{\%} compared to CLHAIF).",
}
```