def en_vgg16(img_size=64):
    from dataset import en
    from models.vgg16 import augmented_vgg16

    model_weights_path = 'models/weights/en_bottleneck_model.h5'
    num_classes = len(en.LABELS)
    model = augmented_vgg16((img_size, img_size, 3), num_classes,
                            weights_path=model_weights_path)
    return img_size, en.LABELS, model
