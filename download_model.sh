#!/bin/bash

# From https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99
download_from_google_drive() {
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
}

if [[ ! -d models ]]
then
    mkdir models
fi

download_from_google_drive "1XiCj0OXZtBfxKyhMEPodt6wp2Jpc3RI9" "model.tar" \
    && tar -xvf model.tar -C models \
    && rm model.tar
