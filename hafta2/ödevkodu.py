# AND Kapısı Girdi ve Hedefleri: (x1, x2, hedef)
veriler = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1)
]

# Ağırlıkları ve bias değerini elle rastgele / keyfi giriyoruz
w1 = -0.7
w2 = -0.3
b = 0.1
eta = 0.2  # Öğrenme katsayısı

epoch = 0

# Tüm örneklerdeki toplam hata 0 oluncaya kadar döngü devam eder
while True:
    epoch += 1
    toplam_hata = 0
    
    # Batch size = Epoch size (4 örneğin tamamı bir turda işlenir)
    for x1, x2, hedef in veriler:
        # 1. Net girdi hesabı
        net = (w1 * x1) + (w2 * x2) + b
        
        # 2. Aktivasyon: net >= 0 ise 1, değilse 0
        cikti = 1 if net >= 0 else 0
        
        # 3. Hata hesabı
        hata = hedef - cikti
        
        # 4. Ağırlık ve bias güncelleme
        w1 += eta * hata * x1
        w2 += eta * hata * x2
        b += eta * hata
        
        # Mutlak hatayı topla (0 olup olmadığını kontrol etmek için)
        toplam_hata += abs(hata)
        
    print(f"Epoch {epoch} -> Toplam Hata: {toplam_hata} | w1: {w1:.2f}, w2: {w2:.2f}, b: {b:.2f}")
    
    # Denklem / model tüm durumlarda hatasız sonuca ulaştıysa dur
    if toplam_hata == 0:
        print(f"\nModel {epoch}. Epoch'ta hatayı 0 yaptı ve öğrendi!")
        break
