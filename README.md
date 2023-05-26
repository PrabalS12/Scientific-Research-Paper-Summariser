# Research Paper Summarizer & Notes Builder

** A lot of extra functionalities are under development process **

Developed a BART Large Language Model for topic-wise accurate and contextual abstractive summaries of the Scientific Research Papers aimed at reducing approximately 70% reading time, with the motive to streamline academic research among students and working professionals.

Fine-tuned pre-trained transformer such as BERT on SciSummNet and BBC News Dataset for context rich and domain specific summaries.

Created extractive summaries using TF-IDF, Sentence Ranking, and Sentence Embeddings to handle large text documents for abstractive summarisation, to produce relevant summaries.

Built an end-to-end summarisation platform with frontend interface for PDF upload and Summary View, Node.js backend script for API and text parsing along with a python script that processes and cleans the textual data and fixes the parsing anomalies before modelling.

A dynamically updating NoSQL database is setup with raw text, domain-specific summaries, and feedback for further re-training and enrichment for domain-specific specialisation.
