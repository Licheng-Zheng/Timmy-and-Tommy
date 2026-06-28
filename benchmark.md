#### Benchmarking & Evaluation Methodology
The automated testing pipeline and evaluation frameworks used to validate the performance, routing accuracy, and generative quality of the Timmy & Tommy architecture.

##### Evaluation Framework
Performance is evaluated using clinical NLP (natural language processing) datasets sourced from the [BRIDGE Benchmark](https://huggingface.co/datasets/YLab-Open/BRIDGE-Open) (Wu, J., Gu, B., et al., *Nature Biomedical Engineering*, 2026). 

**Important Note**: My current system is a RAG system rather than just an LLM (some of their evaluated models are just LLMs). This means my model will/should perform better than just an independent model. The current results are evaluated using the following model: `Llama-4-Scout-17B-16E-Instruct` which scored 71.3% (72.18 with chain of thought reasoning and 60.39 on the few-shot) on the MEDIQA 2019-RQE in zero shot evaluation (the test that 5 questions were used to evaluate my RAG system)
- The model is already pretty good at the test administered, so improvement over the base model is the important metric to consider

##### Testing Pipeline
Architecture relies on an external API (Groq API), so I can't "plug" my model into the test. I need a custom script that sends the payload to Groq, receives the information, and then answers the test. 

For this reason, testing is conducted by an autonomous Python benchmarking script (`benchmark.py`). 

###### Execution Flow:
1. **Data Ingestion:** The script loads the target dataset (e.g., JSON or `.SFT` format) and isolates the instruction, input query, and expected output label.
2. **Payload Construction:** Maps the data to the `ChatRequest` class schema expected by the FastAPI `/api/chat` endpoint.
3. **Live API Sending:** Queries are fired sequentially to the active backend server.
4. **Automated Grading:** The generated response is caught and checked against the dataset's ground truth using `scikit-learn` (for classification) and `rouge_score` (for generation).

##### Engineering Constraints & Solutions

###### Groq API Rate Limiting
On the free plan, there are a maximum of 30 requests per minute. To ensure I get a response on the requests, I retry multiple times after 60 seconds (when it refreshes). There is also a period of time in between requests. 

###### Data contamination 
If my LLM has already trained on the dataset in the past, it may have memorized the answer to the dataset. I have no control over how the model being used is trained (because I'm not the one training), so in the future I will be using the 5-gram truncation test. 
The current model was not evaluated using this framework because it is just a test to make sure my backend is working properly. 
