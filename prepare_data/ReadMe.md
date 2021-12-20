Scripts used to prepare hotel train/dev/test data
Alternatively, email me if you want the files

1. obtain labeled data from https://people.csail.mit.edu/yujia/files/r2a/data.zip
2. obtained unlabeled data from http://times.cs.uiuc.edu/~wang296/Data/LARA/TripAdvisor/TripAdvisorJson.tar.bz2
3. extract all data to ./data/
4. python clean_loc_data_dev.py
5. python clean_loc_data.py
6. shuf ./data/location.train > ./data/location.train.shuf
   (this can create a difference between what is reported in the publication.)
7. python make_allhotel.py
8. python make_embedding_dict.py
9. python split_train.py ./data/location.train.shuf 0