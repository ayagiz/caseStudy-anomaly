1. Öncesinde üniversite bitirme projesinde ile birlikte yapay zeka tecrübesi edinmiştim fakat hemen hemen her şeyi unuttuğumu fark ettim.
2. chatgpt yi bir yol gösterici olarak kullanmaya karar verdim çünkü öğrenmem gerekenler ne bilmiyordum.
3. Projenin öncelikle her sistemden ulaşılabilir olması gerekiyordu. Streamlit ile github ı birbirine bağlayınca bu problem çözülmüş oldu.
4. Streamlit ile beraber projenin deploy edilmiş halde url ile açılabilir olması kullanıcı açısından çok rahat olacaktır.
5. Kullanılabilirlik problemini çözdükten sonra geriye hangi yapay zeka modelini kullanmam gerektiği kalıyordu.
6. Train data sı olarak sadece normal data verildiği için Patchcore son derece uygun bir çözüm olarak göründü çünkü Patchcore zaten sadece normal data çalışan bir model ve eğitilmesi de gerekmiyor. Yani Unsupervised anomaly detection için uygun. PatchCore adı üstünde resimleri küçük bölgelere ayırıp birer patch olarak değerlendirdiği ve puan verdiği için bunun üzerinden heatmap çıkarmasıda kolay olacağı belliydi.
7. Hangi yapay zeka modellerini elediğimize gelecek olursak:
    a. Autoencoder tabanlı yöntemler, eğitimi uzun. RAM streamlit ücretsiz versiyonda kısıtlı olduğu için zaten problemler yaşandı. Resimde ki küçük anomalileri kaçırma riski yüksekmiş, resmi reconstruct ederek hataları blur etmesinden kaynaklı.
    b. GAN-based anomaly detection, modelleri çok büyük olduğu için zaten imkansıza yakın cpu ile eğitilecek bir model için.
    c. Fully supervised classification, train dataların label yok. hepsi normal resim.
    d. Vision Transformer tabanlı büyük modeller, bunlar hakkında bilgim zaten yok. Memory için çıkaracağı problemlerden yine elenmiş oldu.
8. PatchCore için memory bank oluşturma sırasında yaşanılan sorunlara gelecek olursak, Resnet50, patch sayısı 100 ve image size 224 olarak başlanılan projede RAM yetersizliğinden dolayı train sırasında erken terminate ediyordu program. Bunun üzerine resnet18, patch 30, image size 160 düşene kadar modeli indirgemek gerekti. Patchler tek tek .pkl dosyasına yazılabilinirdi fakat günün sonunda prediction yaparken yine bütün .pkl dosyasını
array içine atmak gerekeceği için bu sefer prediction da ram yetersizliği olacaktı.
9. Onun dışında karşılaşılan problemler hep UI case leri üzerineydi. Bu problemlerin çözümü streamlit üzerinden erişilen UI yı daha rahat kullanılabilir hale getirdi.
10. Memory bank oluşturma aşamasından sonra modeli test ederken anomali score larına göre kendi elimle threshold tanımladım. Hazelnut için 4.7 ve wood için 3.6 uygun göründü. Wood seçme sebebim hem data set inin diğer setlere göre büyük olmasının doğru prediction yapmayı kolaylaştıracak hemde hazelnut shape-oriented bir yapı iken WOOD resimleri texture-oriented dı. Modelin 2 durumda da başarılı olup olamayacağını merak etmiştim.
11. Projeye baştan başlasam farklı yapabileceğim bir şey yoktu açıkçası. Yapay zeka konusu üzerine bilgim zaten çok kısıtlı ama öğrenmek istiyorum çünkü bir yazılımcı
için öğrenmesi şart bir araç bana göre. Sistem beklentimin üzerinde çalıştığı için (Hazelnut AUROC = 0.9868 ve Wood = 0.9754) farklı bir şey yapmaya gerek olduğunu düşünmedim. Belki Resnet18 in layer 3 yerine 
layer 2 feature map kullanmış olsam gözle bile görülmesi zor olan anomaliler tespit edilebilinirdi ama yinede garantisi yoktu. Son olarakta image size ı 224 ten 160 a düşürmek RAM kullanımı açısından çokta önemli olmadığını fark ettim. Daha büyük feature map çıksa bile zaten üzerinden 30 tanesi sample edilecekti. Belki de küçük anomalileri sistem bu yüzden gözden kaçırdı.


