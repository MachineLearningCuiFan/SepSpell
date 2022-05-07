

## model_name_or_path  为训练好的探测模型参数地址

accelerate launch test_detection_model.py \
  --test_file data/csv_file/test_sighan15.csv \
  --error_value 0.5 \
  --model_name_or_path "model_output/ChineseBERT_ForCSC_Detection" \
  --batch_size 100 \
  --pad_to_max_length False \
