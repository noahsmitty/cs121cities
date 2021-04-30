# Where am I?

This is a city identifier project we created for our CS121 class software development class. We built an app that is able to identify which city you give it. It is currently only able to accept skyline or landmark images of 5 cities: London, New Delhi, New York, San Francisco, and Tokyo. The app is currently deployed using Flask and Google Cloud Platform at this link: https://cs121-project-311817.wn.r.appspot.com/


### The Model
Our project was created using three ResNet-34 CNN models via fastai. The first model determines wheter the input image is a skyline or landmark image. The other two take a skyline or landmark image and predict which city is pictured. All models were trained on imgaes from Google Images that we then filtered manually for quality.

### Known Bugs
* The website generates an error after uploading around 5 images.
* An unchecked error is generated when a non-image files that has a .jpg, .jpeg, or .png extention is uploaded.

### Acknowledgements
We want to thank Professor Elizabeth Sweedyk for the support she has provided us and Carson Herness, our grutor, for all the assistance with deploying our application.

### Contributors
[Josh Cheung](https://github.com/jcheung-0)

[Noah Smith](https://github.com/noahsmitty)

[Arun Ramakrishna](https://github.com/arunramakrishna)

[Giovanni Castro](https://github.com/gcastro1)

[Chuksi Emuwa](https://github.com/Chuksi101)
