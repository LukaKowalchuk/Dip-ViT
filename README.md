# Dip-ViT-forensically-relevant-insect-classification-model
This repository features all the code used for the Research Project on: A Comparative Evaluation of Vision Transformer and Convolutional Neural Network Performance for Forensically Relevant Dipteran Classification


Author: Luka Kowalchuk

University: University of Amsterdam

Year: 2025-2026

Studentnumber: 13326015

This work aimed to support forensic entomology investigations by accelerating insect identification, a key component in postmortem
interval estimation and wildlife crime investigations. The main backbone architecture
that was implemented is a model called BioCLIP, which is a ViT that is pretrained on
images of the whole biological domain. This model was then finetuned using a Low
Rank Adaptation with a database developed by the Wildlife Forensic Academy, featuring
the specific insects relevant to the forensic entomologist. The research demonstrated
that Vision Transformers outperformed a previously developed Convolutional Neural
Network, while also identifying important limitations in the existing insect database.
Additionally, I explored explainable AI techniques to improve model interpretability and
developed a Streamlit-based demonstration application to illustrate potential real-world implementation.


**Overview of Notebooks and Scripts**

3 Notebooks are featured: dip_vit_updated.ipynb, is used for BioCLIP and BioCLIP-2, vit_16.ipynb is used the for the ViT/B-16, and efficientnetb0_triplet.ipynb is used for the CNN

The code can be run using Jupyter Notebook, where the cells feature discriptions of their contents, how to prepare the dataset, and how to asign directories

The following packages are required to run the code (also featured in the ipynb documents): 

pip install open_clip_torch peft umap-learn plotly scikit-learn tqdm torch torchvision pillow opencv-python numpy matplotlib


Note that some versions may need to be updated/upgraded, as this may resolve certain errors. Running it in a virtual environment is recommended:

python -m venv env

source env\Scripts\activate

The Notebooks are designed to run as a whole, but certain steps can be skipped if for instance a directory with augmented images is intended to be loaded in without
unzipping the full dataset or if certain plots are not of interest.


**Scripts**

Some additional scripts are provided to run the experiments described in the Thesis

These include:

-

-




**Contact**

luka.kowalchuk@student.uva.nl

https://www.linkedin.com/in/luka-kowalchuk-186092222/ 




