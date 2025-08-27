# Mosquito-Species-Classification-Based-On-Acoustic-Signals-On-Microcontrollers
This project was part of my TRIL fellowship at ICTP, sponsored by KAUST. I only created this GitHub account recently, so I won’t upload all the code I wrote — just the final model along with its training and deployment pipeline.

### Helper methods
There are a lot of helper notebooks I wrote, including code to clean the dataset, segment recordings in parallel, and optimize preprocessing before segmentation. I’m skipping those for now and assuming the existence of segmented pickle files or segmented audio files in a directory. If you need any of them, just contact me :)

### Manual DSP implementation
This project includes a manual implementation of Mel-filterbank energy extraction in both Python and C++, with cross-language consistency — which is a known problem in the TinyML community. It’s WICKED FAST (for a mel-spectogram) on any board with an ARM Cortex processor. In my tests (on the Arduino Nano BLE 33 Sense), it performed better using floating-point operations than fixed-point arithmetic, thanks to the insanely optimized libraries for ARM Cortex chips. However, it’s worth noting that performance was worse on the Portenta H7 board for some reason... 

You’ll find a version of the DSP pipeline with a static buffer in C++ to demonstrate matching outputs, along with the code for the triangular filters in Python.

### Model Training code
Of course, I’ve included the training code in Python for the main CNN model. I experimented with a raw audio transformer model, which didn’t perform as well as the CNN + Mel-filterbanks — similar to the results in the TinyChirp paper, so that’s understandable. the transformer outperformed a raw audio CNN. I’ve tried many models, training procedures, and techniques — the notebook contains the one I found to be the most effective.

### Deployment code
You’ll also find a deployment pipeline in C/C++ and Arduino to unite'em! The entire pipeline uses fixed-point arithmetic — except for the manual DSP method, which you can easily swap out for a fixed-point version if needed. I tweaked the keyword detection example from the TinyML TFLite Micro GitHub repo for deployment.

It’s not the cleanest code, since I didn’t originally plan to publish it — it’s just stuff I wrote for myself. But if you’re working on something similar, you’ll probably find some clever tricks in there, especially around Mel-filterbank energy extraction, which you can reuse in your own project no matter what you're building.



