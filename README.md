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

Today I asked claude:

    Verify that the code in the notebook persona_vectors_2.ipynb accurately portrays and reflects the code in the persona_vectors     
    repository. The generated code in persona_vectors_2.ipynb MUST behave exactly the same as the code in this repository. If any     
    differences are found, then generate a new notebook called persona_vectors_3.ipynb that resolves these differences. 


It generated persona_vectors_3.ipynb. 