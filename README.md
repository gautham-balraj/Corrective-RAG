# Corrective RAG Framework with Groq API

## Overview

This repository contains the implementation of Corrective RAG (CRAG), a framework designed to enhance the accuracy, reliability, and robustness of text generation models by incorporating corrective actions and integrating external knowledge sources. The implementation utilizes the Groq API for efficient querying and retrieval of relevant information.

## What is Corrective RAG?

Corrective RAG is a method used to grade documents based on their relevance to a given query or data source. If the retrieved documents are deemed accurate, they undergo refinement to extract more precise knowledge strips. However, if the documents are inaccurate or irrelevant, CRAG resorts to large-scale web searches to find complementary knowledge sources for corrections. The framework ensures that generative models receive refined and accurate information for text generation.

## Workflow

### Retrieval Evaluator

Before utilizing retrieved documents, CRAG employs a retrieval evaluator to assess the overall quality of the information. This evaluator helps determine the relevance and reliability of the retrieved documents for a given query.

### Knowledge Retrieval Actions

Based on the assessment by the retrieval evaluator, different knowledge retrieval actions are triggered:
- **Correct:** If the retrieved documents are deemed accurate, they undergo a refinement process to extract more precise knowledge strips.
- **Incorrect:** In cases where the retrieved documents are inaccurate or irrelevant, they are discarded, and CRAG resorts to large-scale web searches to find complementary knowledge sources for corrections.

### Generative Model Integration

After optimizing the retrieval results through corrective actions, any generative model can be adapted to generate the final text output. CRAG ensures that the generative model receives refined and accurate information for text generation.

### Plug-and-Play Adaptability

CRAG is designed to be plug-and-play, meaning it can be seamlessly integrated into existing Retrieval-Augmented Generation (RAG) frameworks. It has been experimentally implemented with standard RAG and Self-RAG models, demonstrating its adaptability and effectiveness in improving text generation performance across various datasets and tasks.

## CRAG Compared to RAG

While RAG focuses on integrating external knowledge into the generation process, CRAG takes a step further by evaluating, refining, and integrating this knowledge to improve the accuracy and reliability of language models.

## Acknowledgments

Special thanks to the creators of Corrective RAG and the developers of Groq API for their valuable contributions to this project.

---

By incorporating Corrective RAG into your text generation workflows, you can ensure that your models produce more accurate and reliable outputs by leveraging relevant knowledge effectively. For more details on implementation and usage, refer to the code repository.
