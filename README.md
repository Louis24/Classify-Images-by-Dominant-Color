# Classify-Images-by-Dominant-Color

## Task
Classify images by dominant color in one of the following groups: 1. Red 2. Green 3. Blue 4. Other. Ignore grey scale colors.  

## Equation
x=x/(total-grey) where x stand for no. of R,G,B pixel in image, total is size of image and grey for no of scale colors.  

## Requirements
    pip install -r pip-requirements.txt
    
## Usage
Put input folder in the root folder and simply run:
 
    python Classification.py
    
## Results
![](https://github.com/Louis24/Classify-Images-by-Dominant-Color/blob/master/src/red.PNG)
![](https://github.com/Louis24/Classify-Images-by-Dominant-Color/blob/master/src/green.PNG)
![](https://github.com/Louis24/Classify-Images-by-Dominant-Color/blob/master/src/blue.PNG)
![](https://github.com/Louis24/Classify-Images-by-Dominant-Color/blob/master/src/other.PNG)

## Limitation
i. There is a resize operation in preprocessing step, the resized image is 10*10.  
since we do not care about the features in image, Only RGB is important.  
Some of R,G,B maybe classify to "other" due to compress pixels reasons.  
ii. Looking into the equation:  
x=x/(total-grey)  
There is a situation that image only contain grey scale color. So the equation is modified to:  
x=x/(total-grey+1)  

## Cites
This project was inspired by the Author "The Coding Train"  
Here is the youtube playlist  
https://www.youtube.com/playlist?list=PLRqwX-V7Uu6bmMRCIoTi72aNWHo7epX4L  
Also thanks to Andrea Santoro for the API  
https://github.com/dedo24397/ColorClassifier  
https://www.youtube.com/channel/UCTCIs0FhvaMI-0bp-wJaN3A  



