1. Evaluate Qwen3-VL-2B on AMBER, MMHalBench, ObjectHalBench, POPE. Try not to be too detailed, just get an understanding first, look at the scores, observe the mistakes using a targeted approach.

  * Clone Qwen3-VL-2B to current env [DONE]

  * Need to move the cache directory to the current directory.
    `export HF_HOME="/home/users/ntu/hilarius/scratch/fyp"` [DONE]
  * Make a new bash script to ensure the variable persists and inference is done on GPU.

    * Make a new job for this. [DONE]
    * Install flash attn 2.7.2 - cu11 torch 2.6, abi FALSE, cp9
      [https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu11torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl](https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu11torch2.6cxx11abiFALSE-cp39-cp39-linux_x86_64.whl) [DONE]
   

* Evaluate on these benchmarks, understanding the metrics and look at the cases where it failed.

* Remember to use wandb for monitoring.
	 * Evaluate on AMBER
		- Git clone AMBER [DONE]
		- Install dependencies:
			- spacy [DONE]
			- nltk [DONE]
		- Evaluate only on 1004 images generative query and compute score. 
		- Have a bit of understanding on how the metrics are computed. [DONE]
			- Run this inference py file on a single example.  [DONE]
			- Pick one image: [DONE]

get path of image /home/users/ntu/hilarius/scratch/fyp/AMBER/image/AMBER_1.jpg


[
{"id":1,
"response":"This is a photograph of a group of three people walking on a path through a lush, green field on a bright, sunny day.\n\n- **The People**: The three individuals are seen from behind, walking away from the camera. They are dressed in casual, light clothing suitable for a warm day.\n    - The person on the left is a woman wearing a grey t-shirt, light-colored shorts, and a hat.\n    - The person in the middle is a man in a white t-shirt and dark shorts.\n    - The person on the right is a man in a blue t-shirt and brown shorts.\n- **The Environment**: The"}]

List of nouns in the VLM answer
'photograph', 'group', 'people', 'path', 'lush', 'field', 'day', '*', 'People', '*', '*', 'individual', 'camera', 'clothing', 'day', 'person', 'left', 'woman', 't-shirt', 'short', 'hat', 'person', 'middle', 'man', 't-shirt', 'short', 'person', 'right', 'man', 't-shirt', 'short', '*', 'Environment', '*', '*']

List of nouns after selection of proper nouns in AMBER keys (after_process_nouns)
'people', 'path', 'individual', 'camera', 'person', 'woman', 'hat', 'person', 'man', 'person', 'man']

Safe words: list of associated nouns that are deemed truthful, i.e. nouns that are in AMBER answer key
['tree', 'leave', 'bush', 'ground', 'people', 'individual', 'man', 'woman', 'boy', 'girl', 'water', 'sea', 'river', 'hill', 'rock', 'stone', 'reef', 'path', 'street', 'ground']

Hallucinated words: list of associated nouns in "hallu" annotation of the answer.
'bill']

Update safe words to also contain nouns in "truth" annotation of the answer.
'tree', 'leave', 'bush', 'ground', 'people', 'individual', 'man', 'woman', 'boy', 'girl', 'water', 'sea', 'river', 'hill', 'rock', 'stone', 'reef', 'path', 'street', 'ground', 'sky', 'forest', 'grass', 'person', 'lake', 'mountain', 'road']


'people', 'path', 'individual', 'camera', 'person', 'woman', 'hat', 'person', 'man', 'person', 'man']  

First cover safe words in the associations list (ground truth), then cover safe words that is in the ground truth itself.

Similarly for hallucination words.

python inference.py --inference_data /home/users/ntu/hilarius/scratch/fyp/AMBER/sample_qwen_response.json --evaluation_type g


python .\amber_eval.py --inference_data sample_qwen_response.json --evaluation_type g

AMBER baseline notes
- list of associated keywords and globally safe words
- use traditional nlp to find association
- extract nouns from VLM response
- extract safe-words (association + ground truth)
- extract hallucination-words (association + ground truth)
- perform exact matching or vector similarity matching 
- compute list of safe words and hallucination
- compute CHAIR, cover, hal, cog

Comments:
- Criticism of the method:
	- dependent on language toolkit, and the list of keywords is rather extensive and manually typed.
	- there's no severity of hallucination in Hal.
	- not yet a hallucination benchmark assessing existence, attribute, and relation on generative tasks.

- Observation of results:
	- VLMs tend to give affirmative responses, not critically thinking the question it's given.
	- hallucination happens at the middle or end of the response.
	- hallucination and object coverage increases as length increases (tradeoff between response quality and accuracy)

- Ablation Study
	- Image resolution increase will improve performance, except for GPT-4V with no significant difference.
	- Scales of LLMs. Might instead indicate architectural (vision/connectivity modules), no consistent pattern.
	- Training data. Misalignment between instruction-following and academic-structured dataset.


2. Use OPA-DPO and try to analyze how it works and whether it fixed the problem.

Choices:

* Size of Qwen3 VL: 2B, 4B, 8B. Decide to choose 2B for simplicity.
* Choose thinking/instruct variant. Decide to choose instruct as we focus on hallucination, not reasoning.
* Tune the hyperparameters for answer generation.

Make sure to always update on GitHub every new progress is made.






