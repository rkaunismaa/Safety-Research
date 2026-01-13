## Tuesday, January 13, 2026

We will be exploring [Bloom](https://github.com/safety-research/bloom)

I cloned the bloom repository, and then ran /init, which scanned the repo and created the CLAUDE.md file. 

I then asked 

    Continue with your exploration of the bloom repository by reading from    
    https://www.anthropic.com/research/bloom and                              
    https://alignment.anthropic.com/2025/bloom-auto-evals/ which provide a    
    detailed explanation of bloom. Then create a basic starter jupyter        
    notebook that provides a basic introduction to what a developer can do    
    with bloom, calling it bloom_intro.ipynb. 

Claude was at first unable to scan the second url and created the first release of bloom_intro.ipynb. I then asked:

    You said the second url is too long for webfetch. How can we fix this?    
    There is a lot of superb information in this second url!  

It asked:

    ● Good question! A few options:

    1. Fetch with curl/wget and save locally, then I can read the file
    2. Use a web search to find key excerpts
    3. Fetch the page in sections if it has anchors

I went with 1, which it parsed into bloom-alignment-blog.txt, and then updated bloom_intro.ipynb to the current version. Amazing!!
