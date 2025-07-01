# Set the path to your batch_viewer.py here
BATCH_VIEWER_PATH="/n/netscratch/sneel_lab/Lab/jgwang/pythiapile/aria2-1.37.0-linux-gnu-64bit-build1/pythia/utils/batch_viewer.py"

# Loop from 500 to 98500 in increments of 500
for X in $(seq 500 500 98500)
do
  # First command: A = X-25, B = X-10
  A=$((X - 25))
  B=$((X - 10))
  SAVEPATH="intermediate_npys_final/step_${A}_${B}"
  echo "Running batch_viewer for A=${A}, B=${B}..."
  python "${BATCH_VIEWER_PATH}" \
    --start_iteration "${A}" \
    --end_iteration "${B}" \
    --load_path "final/document" \
    --save_path "${SAVEPATH}" \
    --conf_dir "utils/dummy_config.yml"

  # Second command: A = X+10, B = X+25
  A=$((X + 10))
  B=$((X + 25))
  SAVEPATH="intermediate_npys_final/step_${A}_${B}"
  echo "Running batch_viewer for A=${A}, B=${B}..."
  python "${BATCH_VIEWER_PATH}" \
    --start_iteration "${A}" \
    --end_iteration "${B}" \
    --load_path "final/document" \
    --save_path "${SAVEPATH}" \
    --conf_dir "utils/dummy_config.yml"
done

echo "All commands completed!"
