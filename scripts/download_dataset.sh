FILE=$1

if [ "$FILE" == "DFO2K" ]; then
  # Download the DIV2K, Flickr2K, OST dataset
  DIV2K_URL="https://huggingface.co/datasets/goodfellowliu/DIV2K/resolve/main/DIV2K_train_HR.zip"
  Flickr2K_URL="https://huggingface.co/datasets/goodfellowliu/Flickr2K/resolve/main/Flickr2K.zip"
  OST_URL="https://huggingface.co/datasets/goodfellowliu/OST/resolve/main/OST.zip"

  DIV2K_ZIP_FILE=./data/DIV2K_train_HR.zip
  Flickr2K_ZIP_FILE=./data/Flickr2K.zip
  OST_ZIP_FILE=./data/OST.zip

  wget -N DIV2K_URL -O DIV2K_ZIP_FILE
  unzip DIV2K_ZIP_FILE -d ./data
  rm DIV2K_ZIP_FILE

  wget -N Flickr2K_URL -O Flickr2K_ZIP_FILE
  unzip Flickr2K_ZIP_FILE -d ./data
  rm Flickr2K_ZIP_FILE

  wget -N OST_URL -O OST_ZIP_FILE
  unzip OST_ZIP_FILE -d ./data
  rm OST_ZIP_FILE
elif [ "$FILE" == "Set5" ]; then
  # Download the Set5 dataset
  URL="https://huggingface.co/datasets/goodfellowliu/Set5/resolve/main/Set5.zip"
  ZIP_FILE=./data/Set5.zip
  wget -N $URL -O $ZIP_FILE
  unzip $ZIP_FILE -d ./data/Set5
  rm $ZIP_FILE
  echo "Available datasets are DFO2K, Set5"
  exit 1
fi