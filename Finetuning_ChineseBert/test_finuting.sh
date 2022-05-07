
accelerate launch test_finuting_chineseBert.py \
  --test_file ../data/csv_file/test_sighan15.csv \
  --model_name_or_path ChineseBERTFinuting_ForCSC_epoch10 \
  --batch_size 100 \
  --pad_to_max_length False \
