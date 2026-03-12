DVC Pipeline for Phoneme ASR Robustness to Noise
This repository contains a pipeline for testing the robustness of Phoneme Automatic Speech Recognition (ASR) systems against various noise profiles. We use DVC (Data Version Control) to manage large datasets and model artifacts, which are hosted on DagsHub.

1. Environment Setup
Clone the repository and install the required dependencies:

Bash
git clone https://github.com/AnningMa/DVC-pipeline-for-phoneme-ASR-robustness-to-noise.git
cd DVC-pipeline-for-phoneme-ASR-robustness-to-noise
pip install -r requirements.txt

2. DVC Authentication
Since the datasets are stored remotely, you need to configure your credentials to pull them.

Obtain your Default Access Token from your DagsHub account (Settings -> Tokens).

Run the following commands to configure your local environment:

Bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user AnningMa
dvc remote modify origin --local password YOUR_DAGSHUB_TOKEN

3. Pull Data and Models
Once authenticated, you can download all tracked files (e.g., audio samples, noise profiles, or pre-trained weights):

Bash
dvc pull -r origin

4. The project is structured as a DVC pipeline. To reproduce the results or run the experiments:

Bash
dvc repro

This will execute the stages defined in dvc.yaml.
