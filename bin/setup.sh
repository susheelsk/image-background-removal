mkdir bin/mobile_net_model
mkdir bin/xception_model
cd bin
wget http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz
wget http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz

tar xvzf deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz -C mobile_net_model --strip=1
tar xvzf deeplabv3_pascal_train_aug_2018_01_04.tar.gz -C xception_model --strip=1

rm deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz
rm deeplabv3_pascal_train_aug_2018_01_04.tar.gz
