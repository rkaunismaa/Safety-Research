# Safety-Research

Let's explore the various repositories from [Safety Research](https://github.com/safety-research).

## Sunday, January 11, 2026

Today I prompted the Claude CLI (using Sonnet 4.5) with the following command:

    Scan the contents of the persona_vectors folder and the archive article https://arxiv.org/pdf/2507.21509 and then generate a      
    jupyter notebook that demonstrates what these 2 resources are expressing. Call the notebook persona_vectors.ipynb. Use the model  
    found at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct to demonstrate these concepts. 

It chewed on this for about a minute, and then cranked out 'persona_vectors.ipynb'. I am simply stunned on how useful this tool is!! 

And yes, this notebook all runs without modifications in one clean pass! Impressive!

After this, I then switched to Claude Opus 4.5, and asked :

    Create a second notebook persona_vectors_2.ipynb, using the model         
    https://huggingface.co/Qwen/Qwen2.5-7B-Instruct, and this time go into    
    deeper details about persona vectors creating visuals with matplotlib. 

And, yeah, it created persona_vectors_2.ipynb. First time I ran it, there was an error in cell 12. I asked Claude to fix it and the fix worked! It now all runs in one clean pass. Nice!


## Monday, January 12, 2026

Today I asked claude (using claude-opus-4-5-20251101):

    Verify that the code in the notebook persona_vectors_2.ipynb accurately portrays and reflects the code in the persona_vectors     
    repository. The generated code in persona_vectors_2.ipynb MUST behave exactly the same as the code in this repository. If any     
    differences are found, then generate a new notebook called persona_vectors_3.ipynb that resolves these differences. 


It generated persona_vectors_3.ipynb. 

I then asked:

    Create persona_vectors_4.ipynb from persona_vectors_3.ipynb by adding detailed comments to the code that explain what the next    
    step is doing. 

It generated persona_vectors_4.ipynb.

## Tuesday, January 13, 2026

Today I am going to shift my focus to [Introducing Bloom: an open source tool for automated behavioral evaluations](https://www.anthropic.com/research/bloom), which is another, newer repository developed by Safety Research, found [here](https://github.com/safety-research/bloom/). 

I will start with creating a new python environment for bloom. And I think I will start by creating a new subdirectory for all thinks Bloom, to keep things separate from the work on Persona Vectors. Hmm come to think of it, I think it would also be best if I moved EVERYTHING in this current directory (the directories persona_vectors, .claude, .git, .personavectors, and all the current root files EXCEPT this README.md file) into a PersonaVectors subdirectory. Gotta wonder if that is gonna break anything, but we shall see ...

Hmm whelp tried that, and yeah, stuff got messed up ... tried running persona_vectors_4.ipynb using .personavectors but it failed ... meh ... gonna roll back to the initial state. 

OK, so I will keep the Persona Vectors stuff where it currently resides, create a new sub-folder for the Bloom resources that I create, clone the [bloom repo](https://github.com/safety-research/bloom) into this sub-folder, exlude this repo from this repo, create a .bloom python environment, and yeah, continue with this investigation into stuff from [Safety Research](https://github.com/safety-research), and launch Claude CLI from this Bloom sub-folder.


## Wednesday, January 14, 2026

Gonna take a look at [circuit-tracer](https://github.com/safety-research/circuit-tracer.git). 

From a new terminal window inside ~/PythonEnvironments/Safety-Research :

1. uv venv .circuit-tracer
2. source .circuit-tracer
3. mkdir CircuitTracer
4. cd CircuitTracer
5. git clone https://github.com/safety-research/circuit-tracer.git
6. cd circuit-tracer
7. uv pip install .

Hmm just noticed this new environment already has [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) Nice!