# Datasets creation and model learning

In this project we use YOLOv8 by [ultralytics](https://github.com/ultralytics/ultralytics)

Firstly, we create dataset based on MNIST

The full code of creation is [here](https://github.com/Ceasari/digits_recognition/blob/main/datasets/Dataset_creation.py)

The easiest way to repeat is to use [colab notebook](https://colab.research.google.com/drive/10IZhA6NowPVrabE8AsJqztSU-ly8Pdy4?usp=sharing)

You will be able to create the dataset and train model just in few steps. 



## Example of dataset

![train imgages](img/train_ex.png)

## Statistic of dataset

![matrix](img/matrix.png)

![matrix](imgdist_digits.png)  ![matrix](img/distr_classes.png)

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/Ceasari/OD_bot/
    ```

2. Install the required libraries:

    ```
    pip install -r requirements.txt
    ```

3. Download the pre-trained model [link](https://drive.google.com/drive/folders/17ha83DuhPzufn5oN54mMY3WVq3UW3u47?usp=sharing) 


    There are three pre-trained models: 
      
    
    * YOLOv8_m trained based on medium model of YOLOv8m
    
    * YOLOv8_l trained based on large model of YOLOv8l
      
    * YOLOv8_x trained based on extra large model of YOLOv8x

4. Save the downloaded model to the [Model](Model)
   
5. Insert telegram API key and path to downloaded model to `config.py`

6. Run `main.py`

## Usage

1. Start a chat with the bot by searching for `@bot_name`.
2. Send the bot a photo with handwritten digits.
3. Wait for the bot to process the image and return the original image with bounding boxes and labels of each detected digit.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
