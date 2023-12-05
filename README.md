# Selective Progressive Prompts: 
## Improve Forward Transfer by Selectively Concatenating Prompts from Prior Tasks in Continual Learning

**Ty Feng, Raj Kunamaneni, Srivatsan Srikanth, Henry Chou**

### Table of contents
* [Introduction](#star2-introduction)
* [What's in this repository](#question-whats-in-this-repository)
* [Installation](#wrench-installation)
* [How to run](#zap-how-to-run) 
* [Contact](#raising_hand-questions)


## :star2: Introduction
We introduce *Selective Progressive Prompts* – a modified version of Progressive Prompts for Continual Learning in language models. In Selective Progressive Prompts, we learn a set of virtual tokens, or ***soft prompt*** ([B. Lester et al., EMNLP 2021](https://arxiv.org/pdf/2104.08691.pdf)), for each incoming task and sequentially concatenate it with previously learned prompts iff the incoming task is similar enough to previous tasks. Otherwise, we learn a separate per-task prompt using prompt tuning ([B. Lester et al., EMNLP 2021](https://arxiv.org/pdf/2104.08691.pdf)). 


### Our Selective Progressive Prompts Approach
Instead of concatenating every prompt from prior tasks like what Progressive Prompts does [Razdaibiedina et al., ICLR 2023](https://arxiv.org/abs/2301.12314), we perform cosine similarity check between each input embedding and the input embeddings from prior tasks. If there is at least one prior input embedding similar enough to the current input embedding, we would use the previously learned soft prompts when learning the current task. Otherwise, we would learn a separate prompt for the current task. We have a similarity_threshold parameter (default=0.7) that specifies when the model should concatenate previous prompts. Our approach should reduce the negative influence of prompts learned from irrelevant prior tasks when learning the current task.

We used the T5 language model for our work. We based our Selective Progressive Prompts implementation on the original Progressive Prompts implementation of Anastasia Razdaibiedina, Yuning Mao, Rui Hou, Madian Khabsa, Mike Lewis and Amjad Almahairi. ["Progressive Prompts: Continual Learning for Language Models"](https://arxiv.org/abs/2301.12314), ICLR 2023.

![Selective Progressive Prompts](/images/Selective_illustration.png)
Figure: *Illustrating our proposed method **Selective Progressive Prompts**. We use a selection criteria to ensure the relevancy of concatenated prompts, which helps with forward transfer learning on a sequence of hetereogeneous tasks.*

Our paper: ![Selective Progressive Prompts Paper](Selective_Progressive_Prompts.pdf)
Our presentation slides: ![Selective Progressive Prompts Presentation](Selective_Presentation.pdf)

## :question: What's in this repository

This is our code structure:

```
|_T5_codebase/
      |_experiment.sh --> Example experiment for the IMDB-Amazon task pair comparing our approach with progressive prompts and prompt tuning
      |_t5_dataset.py --> T5 Dataset class for reading and processing datasets from Razdaibiedina et al., ICLR 2023
      |_t5_continual.py --> Model class for T5 with prompt tuning and continual learning functions. 
                           Contains our Selective Progressive Prompts implementation.
      |_train_t5_cl.py --> Code to run continual learning experiments with T5 
      
|_datasets/src/data/ --> CL datasets from Zhang et. al., 2015
      |_amazon --> Amazon reviews (zip archive, since dataset is not available through HuggingFace datasets)
      (the rest of datasets can be either accessed through HuggingFace or downloaded by instructions below)
```

**Note ([Razdaibiedina et al., ICLR 2023](https://arxiv.org/abs/2301.12314))**: we access most of the datasets for our experiments through HuggingFace datasets, including CL datasets from Zhang et. al., 2015. Since only one CL datasets from Zhang et. al. is not available on HuggingFace - Amazon Reviews, we uploaded its archived train / test data to ```datasets/src/data/amazon/```. To access the rest of CL datasets (Yelp, Yahoo, AG, DbPedia), you can either use their HuggingFace names in our training script or download them from [http://goo.gl/JyCnZq](http://goo.gl/JyCnZq) to ```datasets/src/data/```.

## :wrench: Installation

Our implementation is based on PyTorch and HuggingFace (transformers + datasets). 

Requirements:
* Python 3.8.5
* Pytorch 1.10.0
* transformers 4.20.0
* datasets 2.3.2
* tqdm, sklearn, numpy, pandas

Step-by-step instructions to get you running Progressive Prompts:

### 1) Clone this repository to your local machine:

```bash
https://github.com/ytyfeng/SelectiveProgressivePrompts.git
```  

A folder called ```SelectiveProgressivePrompts``` with all the codebase should appear.

### 2) Install the required packages:

Make sure that you have Anaconda installed. If not - follow this [miniconda installation](https://docs.conda.io/en/latest/miniconda.html).

To run Selective Progressive Prompts code on GPU, make sure that you have a CUDA capable GPU and the [drivers](https://www.nvidia.com/download/index.aspx?lang=en-us) for your GPU are up to date. In our implementation, we used and CUDA 11.0.

You can re-create our conda enviroment from ```environment.yaml``` file:

```bash
cd SelectiveProgressivePrompts
conda env create -f environment.yaml
```

Your conda should start downloading and extracting packages. This can take ~15-20 minutes.

### 3) Activate the environment:

Your environment should be called ```nlp```, and you can activate it now to run the scripts:

```bash
conda activate nlp
```

## :zap: How to run 

For example, to run Selective Progressive Prompts with T5-large on four tasks (IMDb, CB, SST-2 and DbPedia):
```bash
cd T5_codebase
mkdir my_path_to_save_directory

python train_t5_cl.py --task_list imdb cb sst2 dbpedia_14 --select_k_per_class 1000 \
--lr 0.3 --num_epochs 10 --freeze_weights 1 --prefix_len 10 \
--model_name t5-large --early_stopping 1 \
--save_name T5_experiment --save_dir my_path_to_save_directory
--similarity_threshold 0.7 --batch_size 4
```

In the example above, we froze weights and trained a prompt of token length of 10 (per task) for 10 epochs. We also limited data to 1000 samples per class. We set the similarity threshold to 0.7, meaning only prompts from tasks with similarity above this threshold are concatenated using progressive prompts.
For other arguments and their descriptions, please check ```T5_codebase/train_t5_cl.py``` file.


## :raising_hand: Questions
If you have any questions about the paper or code, please contact Ty Feng (tyfeng[at]ucdavis.edu), Raj Kunamaneni (rkunamaneni[at]ucdavis.edu), Srivatsan Srikanth (ssrikanth[at]ucdavis.edu), Henry Chou (hechou[at]ucdavis.edu) or open an issue. 
