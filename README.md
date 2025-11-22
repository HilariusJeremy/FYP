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
		- Have a bit of understanding on how the metrics are computed. [DONE]
			- Run this inference py file on a single example.  [DONE]
			- Pick one image: [DONE]
			- Evaluate only on 1004 images generative query and compute score.
				- Modify the bash script to generate responses to all queries.
					- Load the query-generative file. [DONE]
					- Update per example generation. [DONE]
					- Use wandb for monitoring. [DONE]
					- Confirm that the implementation is working using interactive job. [DONE]
						10 export HF_HOME="/home/users/ntu/hilarius/scratch/fyp"
						11 module load miniforge3
						12 conda activate fyp
						13 cd "/home/users/ntu/hilarius/scratch/fyp"
						14 python "qwen3-vl-2b-instruct-benchmark-infer.py"
				- Use wandb for monitoring. [DONE]
			- Compute the metrics using the inference script. [DONE]
				python inference.py --inference_data qwen3-vl-2b-instruct-responses.json --evaluation_type g [DONE]

			- Baseline score
				- CHAIR 6.5
				- Cover 54.5
				- Hal 36
				- Cog 1.3
			- Interpretation:
				- Hallucination rate per example is relatively low, but the coverage is only 54% of all labelled objects.
				- More than 1/3 of all queries have hallucination.
				- Of all hallucination cases, rather small amount of them are human aligned.

		- Collect cases where the model failed and analyze its failure.
			- Update AMBER metric calculation script to also collect number of hallucinated tokens and coverage. [DONE]
			- Select the top-30 images with highest CHAIR count. [DONE]
			- Perform manual inspection on about 30-40 images. [DONE]
				- First case is rather confusing. How come label objects (ground truth) get extracted as hallucination tokens? [DONE]
				- Reexecute this example, obtain intermediate results such as safe_list, etc. This is in the inference.py script, so we might need to tailor the script this way. But only CPU should suffice. [DONE]
				- Reexecute inference.py for this one example only: id 811. Understand all [DONE]
		- Insights:
			- No attribute, relation, or adjective benchmarking -> VLM responses might be more opinionated.
			- VLMs often include guessing on the context where the image is taken. This might be evaluated as a 'sentiment' of sorts.
			- Extraction of similar tokens is done simply on vector similarity, without paying attention to the context where the sentence is formed, or any visual grounding with the image.
			- Tradeoff between Coverage and CHAIR can be detected from the threshold for similarity. If we want to make them rather orthogonal, another method is preferred.
		
		- Actionable steps:
			- Consider multimodal grounding embeddings, such as CLIP to consult with image.
			- Consider also Multimodal BERT for comprehension.
			- Multimodal hallucination might be considered a multimodal NER task.
			- Evaluation using multiple prompts and multiple image focus might push the performance of VLMs further.

		- Prototyping AMBER:
			- Introduce CLIP as a visual grounding option.	
			- python inference_clip.py --inference_data unittest_id_811.json
			- Error message:

```
(fyp) [hilarius@x1000c0s5b0n0 AMBER]$ python inference_clip.py --inference_data unittest_id_811.json
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
Device set to use cuda:0
  0%|                                                                                                                | 0/1 [00:02<?, ?it/s]
Traceback (most recent call last):
  File "/scratch/users/ntu/hilarius/fyp/AMBER/inference_clip.py", line 438, in <module>
    main(args)
  File "/scratch/users/ntu/hilarius/fyp/AMBER/inference_clip.py", line 221, in main
    if check_synonyms_word(noun, check_word, image, clip, args.semantic_thres, args.clip_thres):
  File "/scratch/users/ntu/hilarius/fyp/AMBER/inference_clip.py", line 70, in check_synonyms_word
    clip_word1_similarity = F.cosine_similarity(text_emb1, image_emb)
``` 

python inference_clip.py --inference_data unittest_id_811.json --evaluation_type g

- [DONE]
- Implement object detector into the pipeline, using association as the label for zero shot object recognition. [DONE]
	- Detect objects in the image
	- For each object, crop it and assign label from asociation -> every object will have its own CLIP embedding.
	- Assign label for the object by using safe_words from the annotated ground truth label.
	- Three way association between noun, candidate words, and image (region)

	python inference_clip.py --inference_data unittest_id_811.json --evaluation_type g

- Make evaluation more efficient with more GPU utilization. 
	- Make inference more efficient for running 30 examples. [DONE]
	- Updated Score:	
		CHAIR:           4.3
		Cover:           55.3
		Hal:             25.3
		Cog:             3.2



		
		- Read 
			- FaithScore 
			- MMHalBench
			- GAVIE
			- HAELM
			- Bingo
			- HallusionBench (independent for answer sufficiency)
			- CCEval (tradeoff coverage vs CHAIR)
			- Hal-eval (event description)
			- CorrelationQA (spurious visual information)
			- VHTest
			


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

python inference.py --inference_data qwen3-vl-2b-instruct-responses.json --evaluation_type g


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
* Dataset is right now only focused on generation tasks, not discriminative.

Make sure to always update on GitHub every new progress is made.






