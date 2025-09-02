# PPRSA: A Plug-and-Play Language Model with Rational Speech Act Inference for Generating Empathetic and Engaging Dialogue Responses

This is the PyTorch implementation of the paper: PPRSA: A Plug-and-Play Language Model with Rational Speech Act Inference for Generating Empathetic and Engaging Dialogue Responses. The code is still in its early stages. I am working on integrading this code with another paper "Applying Emotion-Cause Entailment for Help-seeker Guidance in Emotional Support Conversations". It would be better to wait for the integrated code finished. The generated responses are located in the "generated_responses_txt" file.

## Abstract
Developing an empathetic dialogue system has been a challenge for years. Although most system responses are grammatically correct and convey emotions to some extent, their quality still falls short for human standards. In this study, an integrated system called the Plug-and-Play Rational Speech Act (PPRSA) was developed by integrating a Transformer-based language model with two frameworks, namely a plug-and-play (PP) framework and a rational speech act (RSA) framework, to generate empathetic and engaging responses. The PP framework was used to control the generation through two attribute models, which ensure the empathetic intent and engagement of responses, thereby increasing empathy and relevance in the generated responses. The RSA framework was used to accelerate and improve response generation by reasoning the perturbation variation from the attribute models. It reduced the response generation time by 73.65% and exhibited enhanced performance compared to the plug-and-play language model (PPLM).  EmpatheticDialogues was used as the evaluation dataset. In terms of objective evaluation, the PPRSA-based system accurately provided empathetic intent and engaging responses based on users’ emotions, thereby providing more comfort and support. The system achieved a BERTScore of 0.869, a Distinct-1 score of 5.19, a Distinct-2 score of 31.88, an empathetic intent accuracy of 41.84%, and an engagement accuracy of 72.24%. In terms of human evaluation, the system outperformed other baseline systems in terms of empathy, relevance, and fluency.

## Quick Start
The codes are still pretty messy. Please note that the shell scripts need to be edited based on your chosen model.

### Install
```
conda create -n PPRSA python=3.8 -f environment.yml
conda activate PPRSA
```

### Train the Generation Model
Choose base model from three models: DialoGPT, BlenderBot, and Llama.
```
cd GenerationModel
bash fine-tune.sh
```

### Train the Attriute Models
Choose the encoder from three models: DialoGPT, BlenderBot, and Llama.
```
cd IntentClassifier_head
bash train_Intent_classifier.sh

cd EngagementClassifier_head
bash train_EngagementClassifier_head.sh
```

### Train the Next Intent Prediction Model
```
cd IntentClassifier_head
bash train_Intent_classifier.sh
```
