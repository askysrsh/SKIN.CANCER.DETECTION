# a function given a function, it predicts the class of the image
def predict_image_class(img_path, model, threshold=0.5):
  img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))
  img = tf.keras.preprocessing.image.img_to_array(img)
  img = tf.expand_dims(img, 0) # Create a batch
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  img = tf.image.convert_image_dtype(img, tf.float32)
  predictions = model.predict(img)
  score = predictions.squeeze()
  if score >= threshold:
    print(f"This image is {100 * score:.2f}% malignant.")
  else:
    print(f"This image is {100 * (1 - score):.2f}% benign.")
    
  plt.imshow(img[0])
  plt.axis('off')
  plt.show()
  
  predict_image_class("data/test/melanoma/ISIC_0013767.jpg", m)
  
  predict_image_class("data/test/nevus/ISIC_0012092.jpg", m)
  predict_image_class("data/test/nevus/ISIC_0012092.jpg", m)
