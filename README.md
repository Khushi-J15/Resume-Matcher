#h1 AI Powered Resume Screener

It is basically a resume shortlisting app which deep down works by capturing semantic similarity between words and their context in NLP. 

Hr enters the job desc

which is then converted into vectors
I have a input data comprising of resumes which is also vectorized using tf idf  
term frequency inverse document freqeuncy

term frequency captures most frequent words and inverse doc. frequency captures most unique and rare words which has more weightage on final prediction 

this nlp model captures context and acts as a medaitor between human laguage and machines
next up another algorithm cosine similarity is used to precompute the similarity between given job description and all the input resumes and saves it in the pickle file so that at real time we don't face any latency to perform all these steps iteratively
the pickle file contains the precomputed similarity scores for all the resumes

one more factor is used while making the model which is the use of weighted similarity i.e. of the given 
job desc the model will give more weightage to experience and skills section , this step makes sure 
that the end user gets well classified, tailored resumes out of all .

  At the UI part  I have made it a multipage application by introducing an admin section that provides 
summary of shortlisted candidates using bar charts and pie charts 
