# Kiln Scaling Project

In this repo, we attempt to scale the kilns identification project across all of South Asia.

## Folders 

```Kiln_data```: Folder containing kiln data points divided into positive and negative csv. Each csv 1-column containing multiple examples (use ```.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_BANDS) ``` to get the correct shape). 


```notebooks```: Folder containing all the Jupyter notebook files we use for our experiments. 


```models```: Folder containing each model's historic performance (```./models/{model_name}/history```) and weights (```./models/{model_name}/weights```)


```utils```: Folder containing useful information such as country boundaries, etc. 

