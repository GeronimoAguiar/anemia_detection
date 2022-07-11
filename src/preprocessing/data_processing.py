from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np


def data_augmentation(path_image, save, path_save_images, quantity_of_images):

    '''
    Recebe o caminho de relativo de uma imagem e realizas transformações nos pixels gerando novas imagens

    Args:
        path_image: String -- caminho relativo para a image a ser aumentada
        save: Boolean -- indica se as imagens devem ser salvas
        path_save_images: String -- caminho relativo do diretório para salvar imagens aumentadas
        quantity_of_images: int -- Quantidades de imagens a serem geradas a partir da imagem de origem

    Returns: tem dois possíveis retornos, caso as imagens sejam salvas o retorno é vazio
             caso não a função retorna um tensorflow.python.keras.preprocessing.image.ImageDataGenerator configurado
             para fazer transformações a nível de pixel.
    '''

    image_original = load_img(path_image)
    image_original = img_to_array(image_original)
    image_original = np.expand_dims(image_original, axis=0)

    aug = ImageDataGenerator(featurewise_center=True,
                              samplewise_center=True,
                              featurewise_std_normalization=False,
                              samplewise_std_normalization=False,
                              zca_whitening=True,
                              zca_epsilon=1e-06,
                              brightness_range=(0.5, 1.5),
                              channel_shift_range=0.3,
                              fill_mode="nearest",
                              horizontal_flip=True,
                              vertical_flip=True,
                              rescale=2)

    if(save):
        total = 0

        # construct the actual Python generator
        imageGenT = augT.flow(image_original, batch_size=1, save_to_dir=path_save_images,
                              save_prefix="augmentation", save_format="jpg")
        # loop over examples from our image data augmentation generator
        for image in imageGenT:
            # increment our counter
            total += 1
            # if we have reached the specified number of examples, break
            # from the loop
            if total == quantity_of_images:
                return

    else:
        return aug