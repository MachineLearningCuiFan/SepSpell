

accelerate launch test_SepSpell.py \
  --error_value 0.5 \
  --test_file data/csv_file/test_sighan15.csv \
  --max_length 200  \
  --detection_model_name_or_path "model_output/ChineseBERT_ForCSC_Detection" \
  --model_name_or_path ChineseBERT_ForCSC_epoch10_version3 \
  --batch_size 100 \
  --pad_to_max_length False \
