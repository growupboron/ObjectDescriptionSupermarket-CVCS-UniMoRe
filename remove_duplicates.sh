#!/bin/bash

# Check if the corrupted.log file exists
if [ -f "corrupted_images.log" ]; then
  # Use awk to remove duplicates and overwrite the original file
  awk '!seen[$0]++' corrupted_images.log > corrupted.tmp && mv corrupted.tmp corrupted_images.log
  echo "Duplicates removed from 'corrupted.log'"
else
  echo "File 'corrupted_images.log' does not exist."
fi

