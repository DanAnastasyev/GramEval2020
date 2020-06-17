#!/bin/bash

# From https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99
download_from_google_drive() {
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
}

if [[ ! -d models ]]
then
    mkdir models
fi

declare -A IDS=(
    ["ru_bert_final_model"]="1RpWcC8PGkSO7eduW5DBCS4ygHKjpEC4P"
    ["chars"]="1XbN1hz0xNZ2GNnSnXh5ewHo2TGJZjJwg"
    ["chars_lstm"]="1UUR3tlLwceEtK8bGzMgeSMndJKm_M368"
    ["chars_morph_lstm"]="1XHYoVIZSMiq4llRyO5ONd1ZzYfVQxu6Y"
    ["frozen_elmo"]="12leckypYyO6-88eqx2x0-tFAGN2OxarR"
    ["frozen_elmo_lstm"]="1ocIT5ObAsxKi-dnVkJrzX0D9z04PAdex"
    ["frozen_elmo_morph_lstm"]="16hoeT3izwX1im2vnf3BgRRicPNsdBG_B"
    ["trainable_elmo"]="1RFcme4sVHnVbwiPOOp4BtP4_MbZn2ijC"
    ["trainable_elmo_lstm"]="1lAZezljNEhq-N8R-C2k3TojwoRCtpidc"
    ["frozen_bert"]="1lAZezljNEhq-N8R-C2k3TojwoRCtpidc"
    ["frozen_bert_lstm"]="1OejzGzV_JuNdEAVneahxaLUWJUIUpjre"
    ["trainable_bert"]="1RdcK5ECIjxOZWxZnbH9g1_XbhJNP6K2N"
    ["trainable_bert_lstm"]="1Khvtoo2cYROH0e-wrD97bn5gWx4NE_ru"
    ["trainable_bert_morph_lstm"]="1dmmk5LjecmDRvrpyAnEIZASnDwbAi_Hu"
)

download_from_google_drive ${IDS[$1]} "model.tar" \
    && mkdir models/$1 \
    && tar -xvf model.tar -C models/$1 \
    && rm model.tar
