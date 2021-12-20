This is repo for "Making a (Counterfactual) Difference One Rationale at a Time"

./prepare_data/ has scripts for preparing the training data for the TripAdvisor data. If you want splits for the RateBeer data, send me an email.
./train/ has scripts for training the models.

The rationale model is defined in ./share/rationale_model.py
The Counterfactual Predictor is defined in ./train/CF_Predictor.py

This repository is derivative of other, better rationale repositories. Some of the utilities, like the embeddings format, probably come from them.
  https://github.com/taolei87/rcnn/blob/master/code/rationale/README.md
  https://github.com/RiaanZoetmulder/Master-Thesis/tree/master/rationale 
  https://github.com/code-terminator/classwise_rationale 
  

