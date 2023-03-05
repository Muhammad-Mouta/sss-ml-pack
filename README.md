# sss-ml-pack
The Machine Learning pack for sSs (Smart Survey System): A complete platform for survey posting and taking with smart features.

## What is sSs?
sSs stands for Smart Survey System. It is a complete platform for survey creation, posting and taking. It was orginally developed as a graduation project to address the problem of long, boring and repetetive opinion surveys. 

## Why do we need Machine Learning in the project?
The ML in the project is an 'auto-answer' feature. It is used to facilitate the survey taking process on the respondents without degrading the quality of responses. The idea is to use previous responses of a survey to generate new responses which the respondent has to validate. The respndent is free to keep the answers as generated or to change them, which provides feedback to the generative model and thus trains it.

## What is the ML algorithm used?
It is an algorithm inspired by Anomaly Detection in which you choose the parameters of a probability density function based on previous data. However, instead of using the probability density function to validate new responses, it is used to generate (sample) new responses.

## What PDF is used in the model?
The PDF used in the model is a custom PDF that is the weighted sum of (m) Normal PDFs with constant variance, where (m) is the number of answers in the survey. We call it Multimodal Normal Disribution. The details of the PDF can be found in the Machine Learning chapter of the project's book uploaded to this repo.

## Usage Example.
```python
>>> import numpy as np
>>> from ml_pack.response_measures import MultimodalDistroModel as MDM
>>> mdm = MDM(q=[5, 5, 5])  # Initialize a survey with 3 questions each with 5 answers
>>> mdm.R   # The histogram matrix is initialize to all ones
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]], dtype=int64)
>>> mdm.auto_answer()   # If you auto answer, the generated answers are completely random
array([[3.],
       [5.],
       [1.]])
>>> mdm.auto_answer()
array([[4.],
       [1.],
       [1.]])
>>> R = m = 0          
>>> for i in range(500):
...     R, m = mdm.update_distro([1, 3, 5]) # Insert the answer into the model 500 times
... 
>>> mdm.R   # The histogram matrix changes
array([[501,   1,   1,   1,   1],
       [  1,   1, 501,   1,   1],
       [  1,   1,   1,   1, 501]], dtype=int64)
>>> mdm.auto_answer()   # The answers are still random but biased towards [1, 3, 5] respectively
array([[2.],
       [3.],
       [4.]])
>>> mdm.auto_answer()
array([[1.],
       [2.],
       [5.]])
>>> mdm.auto_answer()
array([[1.],
       [2.],
       [5.]])
>>> mdm.auto_answer()
array([[1.],
       [4.],
       [5.]])

```

For more information about the model and the thought process behind it, you can find in this repo the model's chapter in the project's book.
