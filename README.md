# IndicBART alongside Visual Element: Multimodal Summarization in Diverse Indian Languages

#### Dataset Used: Y. Verma, Anubhav Jangra, Raghvendra Kumar and Sriparna Saha (2023), “Large Scale Multi-Lingual Multi-Modal Summarization Dataset”,17th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2023), May 2–6, 2023, Croatia.

#### Dataset link: https://github.com/Raghvendra-14/M3LS

In today's era of information abundance, the need for sophisticated summarization techniques is more pronounced than ever, particularly in linguistically diverse areas such as India. This GitHub repository presents an innovative solution for multimodal multilingual summarization, seamlessly integrating textual and visual elements to generate concise and coherent summaries.

### Key Features:

Multimodal Approach: Our research addresses the challenge of summarization by incorporating both textual and visual information, resulting in more comprehensive summaries.
Focused on Indian Languages: We concentrate on four prominent Indian languages - Hindi, Bangla, Gujarati, and Marathi - catering to the linguistic diversity of the region.

State-of-the-Art Models: Leveraging the power of pre-trained models such as IndicBART for text summarization and the Image Pointer model for image summarization, we ensure high-quality outputs.

Abstractive Summarization: Employing abstractive techniques, our approach crafts summaries that capture the essence of the input content while maintaining coherence.

User Satisfaction Evaluation: We provide a robust method like the rouge-1, rouge-l, and bleu-1 scores for evaluating the quality of summaries, enhancing the significance and applicability of our work. For the image-pointing classification method, we use the image-pointer method.

### How to Use:

Use model.py to train the model from scratch or fine-tune it. For fusion of features of text and image, you can use the fusion.ipynb file and use the imagepointer.py file to point the best image to that summary. You can also try to recreate our baseline results using the code baseline_with_imagepointer.ipynb.

### Contributions Welcome:
We invite contributions from researchers and developers interested in advancing summarization techniques, particularly in linguistically diverse contexts. Whether it's enhancing existing models, adding support for additional languages, or improving evaluation methodologies, your contributions can help drive the field forward.

### Citation:

If you use our work in your research, please cite our paper:
@misc{Sharma2024large,
title={IndicBART alongside Visual Element: Multimodal Summarization in Diverse Indian Languages},
author={Deepak Prakash, Raghvendra Kumar, Shubham Sharma and Shubham Sharma},
year={2024}
}.
### License:

This repository is released under the [Apache 2.0 License].

### Contact:

For inquiries or collaborations, feel free to contact us at [amnour.rajsubham@gmail.com]
