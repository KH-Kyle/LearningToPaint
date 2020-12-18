# Learning to Paint
This project is inspired by the original [ICCV2019 paper](https://arxiv.org/abs/1903.04411). We built an agent that can draw paintings like how human do, using a neural renderer combined with Deep Reinforcement Learning. It would decompose a given image into strokes and then apply each stroke onto an empty canvas to create a visual simlar output. After we 1) re-implement essential parts and recover the result from the paper, we focus on 2) applying the technique to a Chinese character dataset and improve RL performance; 3) adding a style transfer into the architecture so that the agent can draw image with given style.

VIDEO GOES HERE (probably): Record a 2-3 minute long video presenting your work. One option - take all your figures/example images/charts that you made for your website and put them in a slide deck, then record the video over zoom or some other recording platform (screen record using Quicktime on Mac OS works well). The video doesn't have to be particularly well produced or anything.

## Table of Contents
- [Introduction](#Introduction)
- [Related Work](#related-work)
- [General Approach](#general-approach)
- [Autonomous Chinese Writing Agent](#autonomous-chinese-writing-agent)
    - [Contribution](#contribution)
    - [Results](#results)
    - [Discussion](#discussion)
- [Learning to Paint with Style](#learning-to-paint-with-style)
    - [Discussion](#discussion)
    - [Visual Results](#visual-results)

## Introduction
No matter in which forms of art, it always comprises the process of perception, understanding, and expression. The expression step shows the wisdom and creativity of human beings. Different people have different idea about an object and they would use different styles to express their feeling. Specifically, painting has been a mainstream form of art for decades. Most people spend a huge amount of time to master this skill. It would be an interesting work to teach machines how to paint. *Huang et al.* introduces an idea about employing a neural renderer in model-based Deep Reinforcement Learning to get an agent learn to make long-term plans to decompose images into strokes. The agent also learns to determine the position and color of each stroke. We recovered their work with our own implementation and then extended based on that. 

Since writing Chinese character also requries a stroke by strokes strategy,   we attempted to modify the existing model so that we could train an autonomous writing agent that writes Chinese characters like how a human would write. 

Besides, it would be nice that the agent can draw a given image with different style. Nerual Style Transfer has alway been a popular topic over years. Now, we combine the idea with our model to generate a painted image with desired styling. 

## Related Work

Our work is highly related to [ICCV2019-Learning to Paint](https://github.com/megvii-research/ICCV2019-LearningToPaint) and [PyTorch-Multi-Style-Transfer](https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer). We apply the idea from the former and use partial codes from both repo to train our model.

## General Approach
Here are graphs representing the overall architecture of training and testing. In different sections, we will mention which parts we implemented by ourselves and which parts we adopted original implementation.

![Training](./material/training.png)
![Testing](./material/testing.png)

<ul>
<li> State: The target image, current canvas, and step number
<li> Actor: Generate actions (strokes) based on states
<li> Renderer: Render the new state as the action applied on the current canvas
<li> Discriminator: Generate Wasserstein distance from states to target image. The difference between the state ane new state is the reward for current state. 
<li> Critic: Predict value function for current state
</ul>

## Autonomous Chinese Writing Agent

The code is in _chinese-character_ branch. For this experiment, we implemented the entire learning process. Many parameter changed since the chinese character dataset is greyscale and we only used pretrained renderer. 
### Contribution

How did you decide to solve the problem? What network architecture did you use? What data? Lots of details here about all the things you did. This section describes almost your whole project.

Figures are good here. Maybe you present your network architecture or show some example data points?

### Results

How did you evaluate your approach? How well did you do? What are you comparing to? Maybe you want ablation studies or comparisons of different methods.

You may want some qualitative results and quantitative results. Example images/text/whatever are good. Charts are also good. Maybe loss curves or AUC charts. Whatever makes sense for your evaluation.

### Discussion

You can talk about your results and the stuff you've learned here if you want. Or discuss other things. Really whatever you want, it's your project.

## Learning to Paint with Style 

For this experiment, we modified the original implementation by add the style transfer implementation. We tuned the models based on pretrained weights given by the author.
### Discussion
We applied the style transfer implementation from [this repo](https://github.com/zhanghang1989/PyTorch-Multi-Style-Transfer). It uses the Multi-style Generative Network.

Since we can treat styling information as a code, we just need to think of where we should embed this information into the agent. Of course, it is possible to transfer the entire image with style before or after the entire painting process. However, it lacks the fun of training the agent and see how the original model would perform. Therefore, we suggested three different places to add the styling information. 

<ol>
<li> Embed the infomration in actor to generate styled action and directly apply the styled action in the renderer.
<li> Embed the information in renderer, so the renderer would generate styled state.
<li> Input styled target into the discriminator for calculation.
</ol>

After the brainstorm and experiments, we got some results. The first option did not converage well, it might because of the hyperparam or we did not have a good model architecture. For the second option, instead of training a new renderer with style transfer, we would simplely pass the output of the renderer into the style transfer model. By doing this, it avoids the probability of model divergence and saves time. Training a new nerual renderer requrires high computational power. For the third method, the model did not coverage. We thought it should be the problem of the problem set-up. Since learning from original image but comparing with styled iamge is too hard for model to learn.
### Visual Results
| Target Image | Style Image | Output |
|--------------|-------------|--------|
|![Demo](./material/flowers.jpg)|![Demo](./material/feathers.jpg)|![Demo](./material/flower.gif)|
|![Demo](./material/shenyang.jpg)|![Demo](./material/candy.jpg)|![Demo](./material/temple.gif)|
|![Demo](./material/seattle.png)|![Demo](./material/wave.jpg)|![Demo](./material/seattle.gif)|
## Dataset
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) 

- [Handwritten Chinese Character (Hanzi) Datasets](https://www.kaggle.com/pascalbliem/handwritten-chinese-character-hanzi-datasets)
