# Portfolio
A small portfolio of some of the projects I found extra interesting during my studies. Some projects are individual and other together with other students.  

**Overview:**  
• Autonomous Brio Labryinth Game - *(Computer Vision)*     
• Word-Based Recurrent Neural Network Cooking Recipe Generator - *(Natural Language Processing - RCNN)*  
• Self-flying Airplane using Proximal Policy Optimization - *(Reinforcement Learning - PPO)*    
• Music Genre Classification using Machine Learning - *(CNN)*  
• Image Processing for Improved Bacteria Classification - *(Image Processing)*  

# Autonomous Brio Labryinth Game 
A summer project whith the aim to automate a traditional Brio Labyrinth Game using computer vision, servor motors and a PID-controller. Together with a colleague in the team I worked with the Computer Vision part of the project. It mainly involved extracting live information about the playing field using a camera, and send the transformed data to a rasberry pi control system. The vision part is visualized in the left view of below image.  

**Video**: [Click here to see full demo run!](https://www.youtube.com/watch?v=qAQdGPwQSzg)  

![image](https://user-images.githubusercontent.com/55993222/161958103-f50a1e7e-85f4-46be-9125-0538b53a8a43.png)    



# Word-Based Recurrent Neural Network Cooking Recipe Generator
**Abstract:**  
A project where I implemented and investigated how a word-based RNN (Recurrent
Neural Network) could be used to generate cooking recipes. The idea is since
it is a word-based RNN, and not character-based, the model should be able to
produce meaningful recipes with contextual sentences. A model was trained
using a dataset of approximately 90 000 recipes containing three sections.
"Title", "Ingredients" and "Instructions"… [Read more](https://github.com/emibr678/Portfolio/blob/main/recipe-generator/tdde16-project-recipe-generator-main/TDDE16_project_report_emibr678.pdf)

    **Example Recipe:**  
    **Title**  
    blueberry-banana bread  

    **Ingredients**  
    • Challah bread for pan  
    • 1 loaf brioche, oil-coated fat in your bakery wedges  
    • All-purpose flour  
    • Sugar  
    • 1 1/2 banana  
    • 2 large eggs  
    • 3 to 4 bananas  
    • 1 cup panko  
    • 1/4 cup coarsely chopped dill then herbs  
    • 1/4 cup, discoloration with fresh hand ground ginger  
    • 3 tablespoons ripe bananas  

    **Instructions**
    • Mix eggs, albumen of zest over dough. Mix bananas on top of bread slices. Cover with banana slices. Let rest of proscuitto have absorbed some honey, covered/coated with no-stick cooking spray  
    • Moisten loaf with banana liqueur and tuck in! place the pan down  
    • 27 ounces honey  
    • 1/4 cup canola oil  
    • 1 sheet 10 ounces semisweet chocolate, australian-style lard or peanut butter  

**Source code:** [Click here](https://github.com/emibr678/Portfolio/tree/main/recipe-generator/tdde16-project-recipe-generator-main)  
**Report:** [Click here](https://github.com/emibr678/Portfolio/blob/main/recipe-generator/tdde16-project-recipe-generator-main/TDDE16_project_report_emibr678.pdf)  

# Self-flying Airplane using Proximal Policy Optimization
**Abstract:**  
This report covers the theory and implementation of training an AI-agent to navigate through an environment, reaching checkpoints without crashing along the way. The agent takes the shape of an old fighter jet and everything is visualized in a 3D environment using Unity. The agent is trained using reinforcement learning methods, more specifically using the algorithm called Proximal Policy Optimization which allows for a more stable and consistent training. The results show that Proximal Policy Optimization can successfully train an agent to follow a set of checkpoints in a certain order and that more complex actions and environments are possible to achieve... [Read more](https://github.com/emibr678/Portfolio/blob/main/self-flying-airplane-ppo/TNM095___Project__danhu028__drika827__emibr678_%20(2).pdf)  
  
![image](https://user-images.githubusercontent.com/55993222/161951366-7fd4e2fc-ba29-4e5a-aa2e-57bbd483405a.png)

**Source code:** [Click here](https://gitlab.liu.se/emibr678/tnm095-ai-plane.git)  
**Video:** [https://www.youtube.com/watch?v=CplOO-hsRcI](https://www.youtube.com/watch?v=CplOO-hsRcI)  

# Music Genre Classification using Machine Learning
The aim of this work is to explore different machine learning approaches to music genre
classification. Furthermore, to explore which of the machine learning techniques selected
(CNN, KNN, and SVM) is best suited in regards to the accuracy of predictions. It is also of
interest to analyze the genres to see which are easiest to classify as well as which are harder
3 to classify and why. Furthermore, it is interesting to see which features in the training data
have the highest impact on the classification of genres... [Read more](https://github.com/emibr678/Portfolio/blob/main/music-genre-classification/report_music_genre_classification.pdf)  

![image](https://user-images.githubusercontent.com/55993222/161953636-100d2600-a5e9-4d8f-b22b-38f77a4ffa7a.png)

**Source code CNN:** [Click here](https://github.com/emibr678/Portfolio/blob/main/music-genre-classification/Training_CNNGenreClassifier.ipynb)

# Image Processing for Improved Bacteria Classification - Thesis
**Abstract**  
Classifiers are being developed for automated diagnostics using images of agar plates. Input images need to be of reasonable quality and consistent in terms of scale, positioning, perspective, and rotation for accurate classification. Therefore, this thesis investigates if a combination of image processing techniques can be used to match each input image to a pre-defined reference model. A method was proposed to identify important key points needed to register the input image to the reference model. The key points were defined by identifying the agar plate, its compartments, and its rotation within the image... [Read more](http://www.diva-portal.org/smash/get/diva2:1452060/FULLTEXT01.pdf)  

![image](https://user-images.githubusercontent.com/55993222/161956647-d3ead3e5-b436-4196-b1f9-1ebd6c847d4d.png)  

**Source code:** [Click here](https://github.com/emibr678/Portfolio/tree/main/Image-Processing-Bacteria)
