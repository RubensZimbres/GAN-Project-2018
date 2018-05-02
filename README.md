# Generative Adversarial Networks in Tensorflow - Linux Shell

My GitHub project on GANs to be run via Windows/Linux shell command

Instructions:

```$ git clone https://github.com/RubensZimbres/GAN-Project-2018

$ cd GAN-Project-2018

$ conda install --yes --file requirements.txt

$ python main.py --epoch=4000 --learning_rate=0.0002 --your_login=rubens```

Arguments:  
--epoch: default=5000  
--learning_rate: default=0.0001  
--sample_size: default=60  
--gen_hidden: # hidden nodes in generator: default=80  
--disc_hidden: # hidden nodes in discriminator: default=80  
--your_login: your login in your OS: default:rubens

Tensorboard will start AFTER you close the pop-up (GAN output with MNIST digits)

<img src=https://github.com/RubensZimbres/GAN-Project-2018/blob/master/Pictures/Screen_output.png> 

<img src=https://github.com/RubensZimbres/GAN-Project-2018/blob/master/Pictures/tensorboard.png>
