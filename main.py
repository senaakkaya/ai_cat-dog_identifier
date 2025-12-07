import os
import shutil
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


# Kötü görüntüleri atlamak için ayar
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# --- VERI TEMİZLEME FONKSİYONU ---
def clean_data_directory(main_dir):
    print(f"Temizleme işlemi başlatılıyor: {main_dir}")
    
    # Kedi ve Köpek klasörleri
    for species in ['Cat', 'Dog']:
        path = os.path.join(main_dir, species)
        
        # Eğer klasör yoksa atla
        if not os.path.isdir(path):
            continue
        
        # Klasördeki tüm dosyaları tara
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            
            # Dosya boyutunu kontrol et (0 KB olanlar silinecek)
            if os.path.getsize(file_path) == 0:
                print(f"Sıfır boyutlu dosya siliniyor: {file_path}")
                os.remove(file_path)
                continue
                
            # Dosyanın gerçekten bir resim olup olmadığını kontrol et
            try:
                img = Image.open(file_path)
                # Resim formatının geçerli olup olmadığını da kontrol edebiliriz
                img.verify()
            except Exception as e:
                # PIL dosyayı tanımlayamıyorsa veya bozuksa sil
                print(f"Bozuk dosya siliniyor: {file_path} (Hata: {e})")
                os.remove(file_path)


DATA_PATH = './microsoft-catsvsdogs-dataset/versions/1/PetImages'

# 0. Bozuk dosyaları temizle
clean_data_directory(DATA_PATH)


# Veri ön işleme ve artırma

# Tüm piksel değerlerini 0-1 arasına normalize ediyoruz (255'e bölerek)
# Ayrıca, modelin daha iyi genelleme yapması için veri artırma uyguluyoruz.
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalizasyon
    rotation_range=20, # Rastgele döndürme
    width_shift_range=0.1, # Genişlik kaydırma
    height_shift_range=0.1, # Yükseklik kaydırma
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2 # Resimlerin %20'si doğrulama için ayrılacak
)

test_datagen = ImageDataGenerator(rescale=1./255) # Test verisine sadece normalizasyon yapılır.

# train_generator (Eğitim Verileri)
train_generator = train_datagen.flow_from_directory(
    DATA_PATH, # İndirilen veri setinin gerçek yolu
    target_size=(150, 150), 
    batch_size=32,
    class_mode='binary',
    subset='training' # Bu, ayrılan kısmın eğitim için kullanılacağını belirtir.
)

# validation_generator (Doğrulama Verileri)
validation_generator = train_datagen.flow_from_directory(
    DATA_PATH, # Yine aynı gerçek yolu kullan
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation' # Bu, ayrılan kısmın doğrulama için kullanılacağını belirtir.
)

# Kategori indekslerini görelim (Örn: {'Cat': 0, 'Dog': 1})
print(train_generator.class_indices)



# Model oluşturma

## 1. CNN Modelini Oluşturma
model = Sequential([
    # 1. Evrişim ve Havuzlama Bloğu
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)), 
    MaxPooling2D((2, 2)),

    # 2. Evrişim ve Havuzlama Bloğu
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # 3. Evrişim ve Havuzlama Bloğu
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Özellik Haritasını Düzleştirme
    Flatten(),

    # Yoğun (Dense) Katmanlar
    Dropout(0.5), # Aşırı öğrenmeyi engellemek için
    Dense(512, activation='relu'),
    
    # Çıkış Katmanı: İki sınıflı (Kedi/Köpek) için 1 nöron ve sigmoid
    Dense(1, activation='sigmoid') 
])

## 2. Modeli Derleme (Compile)
model.compile(
    loss='binary_crossentropy', # İki sınıflı problem için
    optimizer='adam', 
    metrics=['accuracy']
)

## 3. Modeli Eğitme
history = model.fit(
    train_generator,
    # steps_per_epoch: Bir epoch'ta kaç adım (batch) işleneceğini belirler. 
    # Toplam resim sayısı (20000) / batch_size (32)
    steps_per_epoch=train_generator.samples // train_generator.batch_size, 
    
    epochs=10, # 10 döngü boyunca eğit
    
    validation_data=validation_generator,
    # validation_steps: Doğrulama için kaç adım işleneceğini belirler.
    # Toplam doğrulama resmi (5000) / batch_size (32)
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Modeli kaydetme komutu
model.save('kedi_kopek_modeli.h5')
print("Model başarıyla kedi_kopek_modeli.h5 dosyasına kaydedildi.")