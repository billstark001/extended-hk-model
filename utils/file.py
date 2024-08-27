import os


def find_and_rename(folder_path: str, suffix: str, rename: str):
  
  for filename in os.listdir(folder_path):
    if filename.endswith(suffix):
      old_path = os.path.join(folder_path, filename)
      new_path = os.path.join(folder_path, rename)

      try:
        os.rename(old_path, new_path)
        print(f"File renamed from {filename} to {rename}")
        return True
      except OSError as e:
        print(f"Error renaming file: {e}")
        return False

  print(f"No file with suffix {suffix} found in {folder_path}")
  return False
